#!/usr/bin/env python3
"""
Critic Agent (Sufficiency Gate) — Logged ItV + VerifyLocate (Scheme 3) + TF NLI (Iter-3 compatible)

MC/QA: keep original ItV + gap VerifyLocate behavior.

TF:
- Always run statement-support Locate to extract top-K candidate spans (K=5).
- Run ONE batched NLI call (gpt-4o-mini) over the K spans:
    output: SUPPORTED / HARD_CONTRADICTION / NOT_ENOUGH_INFO
- TF final_is_sufficient is derived from NLI (not from cross-encoder score).
- Trace includes tf_nli_* fields for orchestrator gating and answer injection.

NOTE:
- Requires orchestrator to pass retriever for VerifyLocate scoring.
- Requires OPENAI_API_KEY for gpt-4o-mini NLI calls.
"""

import os
import time
import logging
import traceback
import json
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from google import genai

# NLTK sentence splitter (English-friendly)
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

from core.trace_logger import (
    make_item_id,
    write_text,
    write_jsonl,
    write_run_meta,
    get_run_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_emb_model: Optional[SentenceTransformer] = None
_oai_client: Optional[OpenAI] = None
_gem_client = None

TF_LOCATE_TOPK = 5
TF_NLI_MODEL = "gpt-4o-mini"


# ----------------------------
# Model/client getters
# ----------------------------
def get_emb_model() -> SentenceTransformer:
    global _emb_model
    if _emb_model is None:
        logging.info("Loading local embedding model for consensus voting...")
        _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb_model


def get_oai_client() -> OpenAI:
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _oai_client


def get_gem_client():
    global _gem_client
    if _gem_client is None:
        _gem_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _gem_client


# ----------------------------
# Prompts
# ----------------------------
SYS_ID_MC = (
    "You judge whether a CONTEXT is sufficient to answer a multiple-choice question. Name the SINGLE most "
    "important piece of information still MISSING from the context needed to confidently determine the correct "
    "option. If the context already contains everything needed, answer exactly 'NONE'. One short phrase only."
)

SYS_ID_TF = (
    "You judge whether a CONTEXT is sufficient to decide if a TRUE/FALSE statement is true or false. Name the "
    "SINGLE most important piece of information still MISSING from the context needed to confidently decide. "
    "If the context already contains everything needed, answer exactly 'NONE'. One short phrase only."
)

SYS_ID_QA = (
    "You judge whether a CONTEXT is sufficient to accurately answer an open-ended question. Name the SINGLE most "
    "important piece of information still MISSING from the context needed to confidently provide a complete answer. "
    "If the context already contains everything needed, answer exactly 'NONE'. One short phrase only."
)


# ----------------------------
# Helpers
# ----------------------------
def _is_empty(x: str) -> bool:
    return (x is None) or (not str(x).strip())


def _is_none_strict(x: str) -> bool:
    return str(x).strip().upper() == "NONE"


def _is_valid_gap(x: str) -> bool:
    t = str(x or "").strip()
    if not t:
        return False
    if len(t) < 3:
        return False
    if t.endswith("..."):
        return False
    return True


def _gap_is_negative(gap: str) -> bool:
    g = (gap or "").strip().lower()
    neg_markers = [
        "does not state",
        "doesn't state",
        "not stated",
        "not mentioned",
        "not explicitly",
        "isn't mentioned",
        "is not mentioned",
        "lacks",
        "lack of",
        "missing:",
        "context does not",
        "the context does not",
    ]
    return any(m in g for m in neg_markers)


def _normalize_tf_label_iter3(label: str) -> str:
    """
    Normalize any TF label to orchestrator Iter-3 protocol.
    """
    lab = (label or "").strip().upper()
    if lab in ("CONTRADICTED", "HARD_CONTRADICTION"):
        return "HARD_CONTRADICTION"
    if lab in ("SUPPORTED", "NOT_ENOUGH_INFO"):
        return lab
    # We don't produce SOFT_CONTRADICTION in this critic; unknown => safest.
    return "NOT_ENOUGH_INFO"


# ----------------------------
# Model calls: Identify
# ----------------------------
def _call_openai(sys_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 40) -> str:
    try:
        c = get_oai_client()
        r = c.chat.completions.create(
            model="gpt-4o",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _call_gemini(sys_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 256) -> str:
    try:
        client = get_gem_client()
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[f"SYSTEM:\n{sys_prompt}\n\nUSER:\n{user_prompt}\n"],
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        txt = getattr(resp, "text", "") or ""
        return str(txt).strip()
    except Exception:
        return ""


# ----------------------------
# VerifyLocate (Scheme 3)
# ----------------------------
def _make_sentence_windows(context: str, window_sents: int = 2, max_spans: int = 120) -> List[str]:
    if not context:
        return []
    try:
        sents = sent_tokenize(context)
    except Exception:
        sents = [x.strip() for x in context.split("\n") if x.strip()]
    if not sents:
        return []
    w = max(1, int(window_sents))
    spans: List[str] = []
    for i in range(len(sents)):
        chunk = " ".join(sents[i: i + w]).strip()
        if chunk:
            spans.append(chunk)
        if len(spans) >= max_spans:
            break
    return spans


def _score_span(retriever, query: str, span: str) -> float:
    try:
        return float(retriever.score_text(query=query, text=span))
    except TypeError:
        return float(retriever.score_text(query, span))
    except Exception:
        return -1.0


def _locate_topk_spans(
    retriever,
    query: str,
    context: str,
    window_sents: int = 2,
    max_spans: int = 120,
    top_k: int = 5,
) -> Dict[str, Any]:
    spans = _make_sentence_windows(context, window_sents=window_sents, max_spans=max_spans)
    if not spans or retriever is None:
        return {"spans": [], "scores": [], "window_sents": window_sents, "n_spans": len(spans)}

    scored = []
    for sp in spans:
        sc = _score_span(retriever, query=query, span=sp)
        scored.append((sc, sp))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, int(top_k))]

    return {
        "spans": [s for _, s in top],
        "scores": [float(sc) for sc, _ in top],
        "window_sents": window_sents,
        "n_spans": len(spans),
    }


def _verify_locate_with_reranker(
    retriever,
    gap: str,
    context: str,
    window_sents: int = 2,
    max_spans: int = 120,
    threshold: float = 0.55,
) -> Dict[str, Any]:
    spans = _make_sentence_windows(context, window_sents=window_sents, max_spans=max_spans)
    if not spans or retriever is None:
        return {
            "present": False,
            "best_score": None,
            "best_span": "",
            "threshold": threshold,
            "n_spans": len(spans),
            "window_sents": window_sents,
        }

    best_score = None
    best_span = ""

    for sp in spans:
        sc = _score_span(retriever, query=gap, span=sp)
        if best_score is None or sc > best_score:
            best_score = sc
            best_span = sp

    present = (best_score is not None) and (best_score >= float(threshold))
    return {
        "present": bool(present),
        "best_score": best_score,
        "best_span": best_span,
        "threshold": threshold,
        "n_spans": len(spans),
        "window_sents": window_sents,
    }


# ----------------------------
# TF NLI
# ----------------------------
def _tf_nli_judge(statement: str, spans: List[str]) -> Dict[str, Any]:
    """
    One batched NLI call over K candidate spans.

    Output schema:
      {
        "label": "SUPPORTED" | "HARD_CONTRADICTION" | "NOT_ENOUGH_INFO",
        "confidence": 0..1,
        "best_span_index": int | null,
        "citations": [str],
        "explanation": str
      }
    """
    client = get_oai_client()

    system = (
        "You are a strict NLI judge for medical QA. "
        "Given a STATEMENT and EVIDENCE SPANS, decide whether the evidence SUPPORTS the statement, "
        "HARD_CONTRADICTION (directly contradicts), or is NOT_ENOUGH_INFO. "
        "Rules: Only use the provided spans. Do NOT use outside knowledge. "
        "If the spans are merely topically related but do not directly support or contradict the statement, choose NOT_ENOUGH_INFO. "
        "Return JSON ONLY matching the schema."
    )

    evidence_block = []
    for i, sp in enumerate(spans or []):
        sp_clean = (sp or "").strip()
        evidence_block.append(f"[SPAN {i}]\n{sp_clean}")

    user = (
        f"STATEMENT:\n{statement.strip()}\n\n"
        f"EVIDENCE SPANS (top-K):\n" + "\n\n".join(evidence_block) + "\n\n"
        "JSON SCHEMA:\n"
        "{\n"
        '  "label": "SUPPORTED" | "HARD_CONTRADICTION" | "NOT_ENOUGH_INFO",\n'
        '  "confidence": number,\n'
        '  "best_span_index": number | null,\n'
        '  "citations": string[],\n'
        '  "explanation": string\n'
        "}\n"
        "Return JSON only."
    )

    resp = client.chat.completions.create(
        model=TF_NLI_MODEL,
        temperature=0.0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    txt = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(txt)
    except Exception:
        data = {
            "label": "NOT_ENOUGH_INFO",
            "confidence": 0.0,
            "best_span_index": None,
            "citations": [],
            "explanation": f"Failed to parse JSON from model output: {txt[:200]}",
        }

    label_raw = str(data.get("label", "NOT_ENOUGH_INFO")).strip().upper()
    # accept legacy "CONTRADICTED"
    if label_raw == "CONTRADICTED":
        label_raw = "HARD_CONTRADICTION"
    label = _normalize_tf_label_iter3(label_raw)

    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    best_idx = data.get("best_span_index", None)
    try:
        best_idx = int(best_idx) if best_idx is not None else None
    except Exception:
        best_idx = None

    citations = data.get("citations", []) or []
    citations = [str(c) for c in citations if str(c).strip()][:5]

    explanation = str(data.get("explanation", "") or "").strip()

    return {
        "label": label,
        "confidence": conf,
        "best_span_index": best_idx,
        "citations": citations,
        "explanation": explanation,
        "raw_json": data,
    }


# ----------------------------
# Markdown formatter
# ----------------------------
def _format_itv_markdown(trace: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Critic ItV Trace\n")
    for k in [
        "run_dir", "ts", "item_id", "question_id", "q_type", "calls_per_agent",
        "none_frac_strict", "empty_frac", "invalid_gap_frac",
        "consensus_gap", "consensus_gap_is_negative",
        "verify_mode", "verify_label", "verify_best_score", "verify_threshold",
        "final_is_sufficient", "final_missing_info",
        "tf_support_verify_label", "tf_support_best_score",
        "tf_nli_label", "tf_nli_confidence", "tf_nli_best_span_index",
    ]:
        if k in trace:
            lines.append(f"- {k}: `{trace.get(k)}`")
    lines.append("\n---\n")
    lines.append("## Question\n```text\n" + (trace.get("question") or "") + "\n```\n")
    lines.append("## Context (FULL)\n```text\n" + (trace.get("context") or "") + "\n```\n")
    lines.append("\n## Identify Votes\n")
    for v in trace.get("votes", []) or []:
        lines.append(f"- Round {v.get('round')} | {v.get('agent')}: ({v.get('label')}) {v.get('text')}")
    lines.append("\n## Consensus details\n```json\n" + str(trace.get("consensus_debug", {})) + "\n```\n")

    if trace.get("verify_mode") == "locate_reranker":
        lines.append("\n## VerifyLocate (gap)\n")
        lines.append(f"- Label: **{trace.get('verify_label','')}**")
        lines.append(f"- Best score: `{trace.get('verify_best_score')}` (threshold `{trace.get('verify_threshold')}`)")
        lines.append("\n```text\n" + (trace.get("verify_best_span") or "") + "\n```\n")

    if trace.get("q_type") == "TF":
        lines.append("\n## TF Locate top-K spans (statement)\n")
        spans = trace.get("tf_support_top_spans", []) or []
        scores = trace.get("tf_support_top_scores", []) or []
        for i, sp in enumerate(spans):
            sc = scores[i] if i < len(scores) else None
            lines.append(f"\n### SPAN {i} (score={sc})\n```text\n{sp}\n```\n")

        lines.append("\n## TF NLI Judge\n")
        lines.append(f"- Label: **{trace.get('tf_nli_label','')}**")
        lines.append(f"- Confidence: `{trace.get('tf_nli_confidence')}`")
        lines.append(f"- Best span index: `{trace.get('tf_nli_best_span_index')}`")
        lines.append(f"- Citations: `{trace.get('tf_nli_citations')}`")
        lines.append("\nExplanation:\n```text\n" + (trace.get("tf_nli_explanation") or "") + "\n```\n")

    return "\n".join(lines)


# ----------------------------
# Main entry
# ----------------------------
def evaluate_sufficiency(
    question: str,
    context: str,
    q_type: str = "MC",
    calls_per_agent: int = 5,
    question_id: str = None,
    retriever=None,
    verify_threshold: float = 0.55,
    verify_window_sents: int = 2,
    verify_max_spans: int = 120,
) -> Tuple[bool, str, Dict[str, Any]]:
    ts = datetime.now().isoformat(timespec="seconds")
    run_dir = get_run_dir()
    item_id = make_item_id(question_id, question)

    trace: Dict[str, Any] = {
        "ts": ts,
        "run_dir": run_dir,
        "item_id": item_id,
        "question_id": question_id or item_id,
        "q_type": (q_type or "QA").strip().upper(),
        "calls_per_agent": calls_per_agent,
        "question": question,
        "context": context,
        "votes": [],
        "none_frac_strict": None,
        "empty_frac": None,
        "invalid_gap_frac": None,
        "valid_gaps": [],
        "invalid_gaps": [],
        "consensus_gap": "",
        "consensus_gap_is_negative": False,
        "consensus_debug": {},

        # Gap VerifyLocate (MC/QA)
        "verify_mode": "",
        "verify_label": "SKIPPED",
        "verify_best_score": None,
        "verify_best_span": "",
        "verify_threshold": verify_threshold,
        "verify_n_spans": None,
        "verify_window_sents": verify_window_sents,

        # TF locate + NLI
        "tf_support_verify_label": "SKIPPED",
        "tf_support_best_score": None,
        "tf_support_best_span": "",
        "tf_support_threshold": verify_threshold,
        "tf_support_n_spans": None,
        "tf_support_window_sents": verify_window_sents,
        "tf_support_top_spans": [],
        "tf_support_top_scores": [],

        # IMPORTANT: default to NEI so it is never empty on crash paths
        "tf_nli_label": "NOT_ENOUGH_INFO",
        "tf_nli_confidence": 0.0,
        "tf_nli_best_span_index": None,
        "tf_nli_citations": [],
        "tf_nli_explanation": "",
        "tf_nli_raw": None,

        "final_is_sufficient": True,
        "final_missing_info": "",
        "md_path": "",
    }

    q_type_u = trace["q_type"]

    try:
        if retriever is None:
            raise ValueError("evaluate_sufficiency requires retriever for VerifyLocate scoring.")

        # Identify stage
        if q_type_u == "MC":
            sys_id = SYS_ID_MC
            user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
        elif q_type_u == "TF":
            sys_id = SYS_ID_TF
            user_prompt = f"Context:\n{context}\n\nStatement:\n{question}"
        else:
            sys_id = SYS_ID_QA
            user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

        invalid_gaps: List[str] = []
        valid_gaps: List[str] = []
        empty_count = 0
        none_count_strict = 0
        n_votes = 0

        for r in range(1, calls_per_agent + 1):
            o = _call_openai(sys_id, user_prompt, temperature=0.7, max_tokens=40)
            g = _call_gemini(sys_id, user_prompt, temperature=0.2, max_tokens=256)

            for agent, txt in (("openai", o), ("gemini", g)):
                n_votes += 1
                if _is_empty(txt):
                    label = "EMPTY"
                    empty_count += 1
                elif _is_none_strict(txt):
                    label = "NONE"
                    none_count_strict += 1
                else:
                    if _is_valid_gap(txt):
                        label = "GAP_VALID"
                        valid_gaps.append(txt.strip())
                    else:
                        label = "GAP_INVALID"
                        invalid_gaps.append(txt.strip())

                trace["votes"].append({"round": r, "agent": agent, "text": txt, "label": label})

        trace["none_frac_strict"] = float(none_count_strict / max(n_votes, 1))
        trace["empty_frac"] = float(empty_count / max(n_votes, 1))
        trace["invalid_gap_frac"] = float(len(invalid_gaps) / max(n_votes, 1))
        trace["valid_gaps"] = valid_gaps
        trace["invalid_gaps"] = invalid_gaps

        # MC/QA early exit
        if q_type_u != "TF" and trace["none_frac_strict"] >= 0.6:
            trace["final_is_sufficient"] = True
            trace["final_missing_info"] = ""
            trace["consensus_debug"] = {
                "decision": "SUFFICIENT_BY_STRICT_NONE_FRAC",
                "none_frac_strict": trace["none_frac_strict"],
                "empty_frac": trace["empty_frac"],
                "invalid_gap_frac": trace["invalid_gap_frac"],
                "n_votes": n_votes,
                "n_valid_gaps": len(valid_gaps),
                "n_invalid_gaps": len(invalid_gaps),
            }
            md = _format_itv_markdown(trace)
            trace["md_path"] = write_text("critic", f"{item_id}.md", md)
            write_jsonl("critic", "critic_traces.jsonl", trace)
            return True, "", trace

        # Consensus gap
        consensus_gap = ""
        if valid_gaps:
            emb = get_emb_model()
            embeddings = emb.encode(valid_gaps, normalize_embeddings=True)
            sim = np.dot(embeddings, embeddings.T)
            mean_sims = sim.mean(axis=1)
            consensus_idx = int(mean_sims.argmax())
            consensus_gap = valid_gaps[consensus_idx]

            trace["consensus_gap"] = consensus_gap
            trace["consensus_gap_is_negative"] = _gap_is_negative(consensus_gap)

            topk = min(5, len(valid_gaps))
            order = mean_sims.argsort()[::-1][:topk].tolist()
            trace["consensus_debug"] = {
                "decision": "CONSENSUS_SELECTED_ON_VALID_GAPS",
                "n_valid_gaps": len(valid_gaps),
                "consensus_idx": consensus_idx,
                "top_mean_sim": [
                    {"idx": int(i), "mean_sim": float(mean_sims[int(i)]), "gap": valid_gaps[int(i)]}
                    for i in order
                ],
            }
        else:
            trace["consensus_debug"] = {
                "decision": "NO_VALID_GAPS_AVAILABLE",
                "none_frac_strict": trace["none_frac_strict"],
                "empty_frac": trace["empty_frac"],
                "invalid_gap_frac": trace["invalid_gap_frac"],
                "n_votes": n_votes,
                "n_valid_gaps": 0,
                "n_invalid_gaps": len(invalid_gaps),
            }

        # Gap VerifyLocate (MC/QA)
        gap_missing = False
        if consensus_gap:
            trace["verify_mode"] = "locate_reranker"
            vr = _verify_locate_with_reranker(
                retriever=retriever,
                gap=consensus_gap,
                context=context,
                window_sents=verify_window_sents,
                max_spans=verify_max_spans,
                threshold=verify_threshold,
            )
            trace["verify_best_score"] = vr.get("best_score")
            trace["verify_best_span"] = vr.get("best_span", "")
            trace["verify_threshold"] = vr.get("threshold")
            trace["verify_n_spans"] = vr.get("n_spans")
            trace["verify_window_sents"] = vr.get("window_sents", verify_window_sents)

            if vr.get("present") is True:
                trace["verify_label"] = "PRESENT"
                trace["consensus_debug"]["verify_decision"] = "PRESENT->SUFFICIENT"
                gap_missing = False
            else:
                trace["verify_label"] = "ABSENT"
                trace["consensus_debug"]["verify_decision"] = "ABSENT->INSUFFICIENT"
                gap_missing = True

        # TF: Locate + NLI
        if q_type_u == "TF":
            loc = _locate_topk_spans(
                retriever=retriever,
                query=question,
                context=context,
                window_sents=verify_window_sents,
                max_spans=verify_max_spans,
                top_k=TF_LOCATE_TOPK,
            )
            top_spans = loc.get("spans", []) or []
            top_scores = loc.get("scores", []) or []

            trace["tf_support_top_spans"] = top_spans
            trace["tf_support_top_scores"] = top_scores
            trace["tf_support_n_spans"] = loc.get("n_spans")
            trace["tf_support_window_sents"] = loc.get("window_sents", verify_window_sents)
            trace["tf_support_threshold"] = verify_threshold

            if top_spans:
                trace["tf_support_best_span"] = top_spans[0]
                trace["tf_support_best_score"] = top_scores[0] if top_scores else None
                trace["tf_support_verify_label"] = "PRESENT"
            else:
                trace["tf_support_best_span"] = ""
                trace["tf_support_best_score"] = None
                trace["tf_support_verify_label"] = "ABSENT"

            nli = _tf_nli_judge(statement=question, spans=top_spans)
            trace["tf_nli_label"] = _normalize_tf_label_iter3(nli.get("label", ""))
            trace["tf_nli_confidence"] = nli.get("confidence", 0.0)
            trace["tf_nli_best_span_index"] = nli.get("best_span_index")
            trace["tf_nli_citations"] = nli.get("citations", [])
            trace["tf_nli_explanation"] = nli.get("explanation", "")
            trace["tf_nli_raw"] = nli.get("raw_json")

        # Final decision
        if q_type_u == "TF":
            nli_label = _normalize_tf_label_iter3(trace.get("tf_nli_label"))
            trace["tf_nli_label"] = nli_label
            sufficient = nli_label in ("SUPPORTED", "HARD_CONTRADICTION")

            trace["final_is_sufficient"] = bool(sufficient)
            if sufficient:
                trace["final_missing_info"] = ""
                is_sufficient, missing_info = True, ""
            else:
                if consensus_gap:
                    trace["final_missing_info"] = f"{question}\n\nHint gap: {consensus_gap}"
                else:
                    trace["final_missing_info"] = question
                is_sufficient, missing_info = False, trace["final_missing_info"]
        else:
            if consensus_gap and gap_missing:
                trace["final_is_sufficient"] = False
                trace["final_missing_info"] = consensus_gap
                is_sufficient, missing_info = False, consensus_gap
            else:
                trace["final_is_sufficient"] = True
                trace["final_missing_info"] = ""
                is_sufficient, missing_info = True, ""

        md = _format_itv_markdown(trace)
        trace["md_path"] = write_text("critic", f"{item_id}.md", md)
        write_jsonl("critic", "critic_traces.jsonl", trace)

        return is_sufficient, missing_info, trace

    except Exception as e:
        # HARDENED: TF must never fail-safe to "sufficient"
        trace["error"] = str(e)
        trace["traceback"] = traceback.format_exc()

        if q_type_u == "TF":
            trace["tf_nli_label"] = _normalize_tf_label_iter3(trace.get("tf_nli_label") or "NOT_ENOUGH_INFO")
            trace["tf_nli_confidence"] = trace.get("tf_nli_confidence", 0.0) or 0.0
            trace["final_is_sufficient"] = False
            trace["final_missing_info"] = question
            trace["consensus_debug"] = {"decision": "FAIL_SAFE_TF_NEI_ON_ERROR"}
            try:
                md = _format_itv_markdown(trace)
                trace["md_path"] = write_text("critic", f"{item_id}.md", md)
                write_jsonl("critic", "critic_traces.jsonl", trace)
            except Exception:
                pass
            return False, question, trace

        trace["final_is_sufficient"] = True
        trace["final_missing_info"] = ""
        trace["consensus_debug"] = {"decision": "FAIL_SAFE_SUFFICIENT_ON_ERROR"}
        try:
            md = _format_itv_markdown(trace)
            trace["md_path"] = write_text("critic", f"{item_id}.md", md)
            write_jsonl("critic", "critic_traces.jsonl", trace)
        except Exception:
            pass
        return True, "", trace


# ----------------------------
# Public helper (for orchestrator)
# ----------------------------
def rerun_tf_nli(
    statement: str,
    context: str,
    retriever,
    window_sents: int = 2,
    max_spans: int = 120,
    top_k: int = TF_LOCATE_TOPK,
) -> Dict[str, Any]:
    loc = _locate_topk_spans(
        retriever=retriever,
        query=statement,
        context=context,
        window_sents=window_sents,
        max_spans=max_spans,
        top_k=top_k,
    )
    top_spans = loc.get("spans", []) or []
    top_scores = loc.get("scores", []) or []

    nli = _tf_nli_judge(statement=statement, spans=top_spans)

    return {
        "top_spans": top_spans,
        "top_scores": top_scores,
        "n_spans": loc.get("n_spans"),
        "window_sents": loc.get("window_sents", window_sents),
        "nli": nli,
    }
