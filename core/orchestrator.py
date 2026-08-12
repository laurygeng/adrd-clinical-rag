#!/usr/bin/env python3
"""
Orchestrator (Unified ItV Architecture)

Key hardening:
- All question types (TF, MC, QA) use the unified Identify-then-Verify (ItV) gap mechanism.
- Removed legacy NLI branching.
- Answers are delegated to the LLM (gpt-4o shim) to evaluate the final unified context.
- Added precise step-by-step execution timers for benchmarking.
"""

import os
import time
import logging
import hashlib
from typing import Dict, Any, List, Union
from datetime import datetime

from openai import OpenAI

from core.advanced_retriever import AdvancedRetriever
from core.critic_agent import evaluate_sufficiency
from core.answer_agent import generate_final_answer
from core.search_agent import research
from core.trace_logger import write_jsonl, make_item_id, get_run_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_retriever_instance = None
_oai_client = None


def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        logging.info("Initializing AdvancedRetriever...")
        _retriever_instance = AdvancedRetriever()
    return _retriever_instance


def get_oai_client():
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _oai_client


def deduplicate_passages(passages: List[str]) -> List[str]:
    seen_hashes = set()
    unique: List[str] = []
    for p in passages or []:
        if not p:
            continue
        h = hashlib.md5(p.strip()[:100].encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(p)
    return unique


def format_evidence_items(evidence_items: List[Union[str, Dict[str, Any]]]) -> List[str]:
    out: List[str] = []
    for ev in evidence_items or []:
        if isinstance(ev, str):
            t = ev.strip()
            if t:
                out.append(t)
            continue

        source = (ev.get("source") or "").strip()
        title = (ev.get("title") or "").strip()
        url = (ev.get("url") or "").strip()
        text = (ev.get("text") or "").strip()
        if not text:
            continue

        header = f"[WEB:{source}]"
        if title:
            header += f" {title}"
        if url:
            header += f" ({url})"
        out.append(header + "\n" + text)

    return out


def sanitize_gap_text(gap: str) -> str:
    if gap is None:
        return ""
    g = str(gap).strip()
    if not g:
        return ""
    return g[:800]


def generate_web_rewrite_query(statement: str, gap_hint: str = "") -> str:
    s = statement.strip()
    gh = sanitize_gap_text(gap_hint)
    
    if not gh:
        return s
        
    client = get_oai_client()
    sys_prompt = "You are an expert medical search query generator."
    user_prompt = (
        f"We are trying to verify this statement:\n\"{s}\"\n\n"
        f"However, the current evidence is missing this critical information:\n\"{gh}\"\n\n"
        "Please generate a single, highly precise Google search query (3-8 keywords) "
        "to find this exact missing information. Focus on specific medical terms or guidelines. "
        "Output ONLY the query text without quotes, markdown, or preamble."
    )
    
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=30,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        query = (r.choices[0].message.content or "").strip().strip('"\'')
        return query if query else f"{s} {gh}"
    except Exception as e:
        logging.warning(f"Query rewrite LLM call failed: {e}")
        return f"{s}\n\nSearch query focus: {gh}".strip()


def run_pipeline(question: str, q_type: str = "MC", question_id: str = "") -> Dict[str, Any]:
    retriever = get_retriever()
    client = get_oai_client()

    run_dir = get_run_dir()
    item_id = make_item_id(question_id, question)

    trace_log: Dict[str, Any] = {
        "run_dir": run_dir,
        "item_id": item_id,
        "question_id": question_id,
        "q_type": q_type,

        "is_sufficient": None,
        "missing_info": "",

        "critic_decision": "",
        "critic_none_frac_strict": None,
        "critic_empty_frac": None,
        "critic_invalid_gap_frac": None,
        "critic_consensus_gap": "",
        "critic_consensus_gap_is_negative": False,

        "critic_verify_mode": "",
        "critic_verify_label": "",
        "critic_verify_best_score": None,
        "critic_verify_threshold": None,
        "critic_verify_best_span": "",
        "critic_verify_n_spans": None,
        "critic_verify_window_sents": None,
        "critic_md_path": "",

        "completion_triggered": False,
        "completion_reason": "",
        "web_query_used": "",

        # 耗时统计初始化
        "time_base_retrieval": 0.0,
        "time_critic_evaluation": 0.0,
        "time_completion_retrieval": 0.0,
        "time_answer_generation": 0.0,
    }

    # STEP 1: Base Retrieval (计时)
    logging.info("Step 1: Running Base Retrieval...")
    t0 = time.time()
    base_passages, _, _ = retriever.get_retrieved_passages(
        question,
        top_k=8,
        bm25_weight=0.3,
        vector_weight=0.7,
    )
    base_context = "\n\n".join(base_passages)
    trace_log["time_base_retrieval"] = time.time() - t0

    # STEP 2: Critic (Unified) (计时)
    logging.info("Step 2: Critic Agent evaluating sufficiency...")
    t0 = time.time()
    is_sufficient, missing_info, critic_trace = evaluate_sufficiency(
        question=question,
        context=base_context,
        q_type=q_type,
        calls_per_agent=5,
        question_id=question_id,
        retriever=retriever,
    )
    trace_log["time_critic_evaluation"] = time.time() - t0

    trace_log["is_sufficient"] = is_sufficient
    trace_log["missing_info"] = (missing_info or "").strip()

    if isinstance(critic_trace, dict):
        consensus_debug = critic_trace.get("consensus_debug", {}) or {}
        trace_log["critic_decision"] = consensus_debug.get("decision", "")

        trace_log["critic_none_frac_strict"] = critic_trace.get("none_frac_strict")
        trace_log["critic_empty_frac"] = critic_trace.get("empty_frac")
        trace_log["critic_invalid_gap_frac"] = critic_trace.get("invalid_gap_frac")
        trace_log["critic_consensus_gap"] = critic_trace.get("consensus_gap", "")
        trace_log["critic_consensus_gap_is_negative"] = critic_trace.get("consensus_gap_is_negative")

        trace_log["critic_verify_mode"] = critic_trace.get("verify_mode", "")
        trace_log["critic_verify_label"] = critic_trace.get("verify_label", "")
        trace_log["critic_verify_best_score"] = critic_trace.get("verify_best_score")
        trace_log["critic_verify_threshold"] = critic_trace.get("verify_threshold")
        trace_log["critic_verify_best_span"] = critic_trace.get("verify_best_span", "")
        trace_log["critic_verify_n_spans"] = critic_trace.get("verify_n_spans")
        trace_log["critic_verify_window_sents"] = critic_trace.get("verify_window_sents")
        trace_log["critic_md_path"] = critic_trace.get("md_path", "")

    # STEP 3-5: Unified Completion Routing (计时)
    final_context = base_context
    critic_verify_label = (trace_log.get("critic_verify_label") or "").strip().upper()
    consensus_gap_is_negative = bool(trace_log.get("critic_consensus_gap_is_negative"))

    should_complete = (not is_sufficient) and (critic_verify_label == "ABSENT") and (not consensus_gap_is_negative)
    
    if should_complete:
        t0 = time.time()
        trace_log["completion_triggered"] = True
        trace_log["completion_reason"] = "critic_insufficient_and_absent"

        logging.info("Step 3: Triggering Completion Retrieval...")
        completion_query = generate_web_rewrite_query(statement=question, gap_hint=trace_log.get('missing_info',''))
        
        gap_passages, _, _ = retriever.get_retrieved_passages(
            completion_query,
            top_k=5,
            bm25_weight=0.5,
            vector_weight=0.5,
        )

        web_passages: List[str] = []
        try:
            web_evidence, web_query = research(
                client=client,
                target_info=completion_query,
                retriever=retriever,
            )
            trace_log["web_query_used"] = web_query or ""
            web_passages = format_evidence_items(web_evidence)
        except Exception as e:
            logging.warning(f"Web retrieval failed: {e}. Proceeding with local gap passages only.")
            trace_log["web_query_used"] = ""

        merged_passages = deduplicate_passages(base_passages + gap_passages + web_passages)
        final_context = "\n\n".join(merged_passages[:15])
        trace_log["time_completion_retrieval"] = time.time() - t0

    # STEP 6: Final Answer (Unified Delegation to LLM) (计时)
    logging.info("Step 6: Answer Agent generating final output...")
    t0 = time.time()
    final_answer = generate_final_answer(
        client=client,
        question=question,
        context=final_context,
        q_type=q_type,
        model_name="gpt-4o",
    )
    trace_log["time_answer_generation"] = time.time() - t0

    try:
        write_jsonl(
            "orchestrator",
            "pipeline_traces.jsonl",
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "item_id": item_id,
                "question_id": question_id,
                "q_type": q_type,
                "trace": trace_log,
                "final_answer": final_answer,
                "base_context_chars": len(base_context or ""),
                "final_context_chars": len(final_context or ""),
                "base_context": base_context,
                "final_context": final_context,
            },
        )
    except Exception:
        pass

    return {"final_answer": final_answer, "final_context": final_context, "trace": trace_log}