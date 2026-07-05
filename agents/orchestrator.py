#!/usr/bin/env python3
"""Orchestrator Agent (agent ①) — deterministic while-loop coordinating the 7-agent system.

Lightweight route (no LangGraph/AutoGen): a plain Python loop + the Blackboard dataclass, so the
control flow is 100% transparent and every gate is ablatable (flip a judge flag, re-run).

Flow (TF path shown; MC keeps the existing comparative generation):
  Answer Agent -> baseline draft (local only)
  [if TF] Adversarial Web-Research -> web_pro / web_con on the blackboard
          Heterogeneous Court (Entity | Modal | Fact judges + veto Arbiter) over local+pro+con
          -> TRUE/FALSE only overrides the baseline; INSUFFICIENT keeps it (churn blocked)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generate"))
from blackboard import Blackboard
import verification_court as court
import adversarial_web as web

def _build_ctx(passages, n=10):
    import generate_answers_gpt4_ADRD_Bench as G
    return G.build_context_from_passages(passages, n)

def run_tf(bb: Blackboard, local_passages, retriever, client, generate_answer,
           enable_web=True, enable_court=True):
    """Executes the TF pipeline over a Blackboard. Returns the final answer ('Yes'/'No')."""
    local_ctx = _build_ctx(local_passages)
    bb.retrieved_evidence["local"] = local_passages

    # Answer Agent — baseline draft from local context
    bb.baseline_answer = generate_answer(client, bb.question, local_ctx, "TF")
    bb.log(f"baseline={bb.baseline_answer}")
    if not enable_web:
        bb.final_decision = bb.baseline_answer
        return bb.final_decision

    # Adversarial Web-Research Agent — dual-personality pro/con
    pro, con, (pq, cq) = web.research(client, bb.question, retriever)
    bb.retrieved_evidence["web_pro"] = pro
    bb.retrieved_evidence["web_con"] = con
    bb.queries += [pq, cq]
    pro_ev = " ".join(pro[:4]); con_ev = " ".join(con[:4])
    if not (pro_ev.strip() or con_ev.strip()):
        bb.final_decision = bb.baseline_answer
        return bb.final_decision

    if not enable_court:
        # ablation: trust the (support) evidence with a plain verdict (no heterogeneous defenses)
        v, _ = court.fact_judge(bb.question, pro_ev)
        bb.final_decision = {"TRUE": "Yes", "FALSE": "No"}.get(v, bb.baseline_answer)
        return bb.final_decision

    # Heterogeneous Court: entity/modal/fact judge the SUPPORT evidence; refutation evidence
    # can only VETO (a concept-swap in pro is caught by entity/modal, not by con).
    decision, flags = court.court(bb.question, pro_ev)
    bb.judge_flags = {"entity_match": flags["entity"] == "ALIGNED",
                      "modal_match": flags["modal"] == "MATCH", "fact": flags["fact"]}
    bb.log(f"court={decision} flags={flags}")
    if decision == "TRUE":
        bb.final_decision = "Yes"
    elif decision == "FALSE":
        bb.final_decision = "No"
    else:                                    # INSUFFICIENT -> veto blocked the override, keep baseline
        bb.final_decision = bb.baseline_answer
    return bb.final_decision
