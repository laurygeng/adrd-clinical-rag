#!/usr/bin/env python3
"""
Critic Agent (Sufficiency Gate) — Semantic Clustering & Comprehensive Trace Logging
"""

import os
import time
import logging
import json
import re
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from core.llm_engine import LocalLLMClient

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
    get_run_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [CRITIC] %(message)s")
logger = logging.getLogger(__name__)

def _dbg_enabled() -> bool:
    return os.environ.get("HULU_DEBUG", "1") == "1"

def _clip_text(x: str, limit: int = 1000) -> str:
    s = str(x or "")
    if len(s) <= limit:
        return s
    return s[:limit] + f" ...[truncated {len(s) - limit} chars]"

def _dbg(stage: str, **kwargs) -> None:
    if not _dbg_enabled():
        return
    payload = " | ".join(f"{k}={_clip_text(v)}" for k, v in kwargs.items())
    logger.warning(f"[HULU-DEBUG][critic][{stage}] {payload}")


_emb_model: Optional[SentenceTransformer] = None
_llm_client: Optional[LocalLLMClient] = None

CRITIC_CALLS_PER_AGENT = int(os.environ.get("CRITIC_CALLS_PER_AGENT", "5"))
CRITIC_LLM_TEMPERATURE = float(os.environ.get("CRITIC_OPENAI_TEMPERATURE", "0.3"))
CRITIC_ENABLE_RESCUE = os.environ.get("CRITIC_ENABLE_RESCUE", "1") == "1"
CRITIC_RESCUE_TRIGGER_FRAC = float(os.environ.get("CRITIC_RESCUE_TRIGGER_FRAC", "0.67"))
DIVERSITY_THRESHOLD = 0.85


def get_emb_model() -> SentenceTransformer:
    global _emb_model
    if _emb_model is None:
        logger.info("Loading local embedding model for consensus clustering...")
        _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb_model

def get_llm_client() -> LocalLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLLMClient()
    return _llm_client


_CRITIC_FORMAT_RULES = (
    "You MUST output your answer strictly as a valid JSON object.\n"
    "CRITICAL: Do NOT write any introduction, conclusion, or explanation. "
    "Your very first character MUST be '{' and your very last character MUST be '}'.\n"
    "{\n"
    '  \"sufficient\": true,\n'
    '  \"gap\": \"NONE\"\n'
    "}\n"
    "OR\n"
    "{\n"
    '  \"sufficient\": false,\n'
    '  \"gap\": \"<short missing fact phrase>\"\n'
    "}\n"
    "CRITICAL RULES FOR GAP IDENTIFICATION:\n"
    "1. Formulate the 'gap' as a concise, DECLARATIVE phrase representing the missing concept, NOT as a question.\n"
    "2. Focus on the single most important missing concept needed to decide the question.\n"
    "3. Do NOT solve the question or mention option letters."
)

SYS_ID_MC = f"You judge whether a CONTEXT is sufficient to answer a multiple-choice question. Identify only whether the context is sufficient and, if not, the single most important missing fact needed to determine the correct option. {_CRITIC_FORMAT_RULES}"
SYS_ID_TF = f"You judge whether a CONTEXT is sufficient to decide if a TRUE/FALSE statement is true or false. Identify only whether the context is sufficient and, if not, the single most important missing fact needed to decide. {_CRITIC_FORMAT_RULES}"
SYS_ID_QA = f"You judge whether a CONTEXT is sufficient to answer an open-ended question completely and accurately. Identify only whether the context is sufficient and, if not, the single most important missing fact needed to answer. {_CRITIC_FORMAT_RULES}"

_RESCUE_SYS = (
    "You are a sufficiency judge. Output EXACTLY two lines and nothing else:\n"
    "sufficient: yes|no\n"
    "gap: NONE|<short missing fact phrase>\n"
    "If you are unsure, choose sufficient: no and provide the single most important missing fact phrase.\n"
    "NEVER say 'I don't know'."
)

_UNKNOWN_PATTERNS = [
    r"\bi\s*don't\s*know\b", r"\bi\s*do\s*not\s*know\b", r"\bnot\s+enough\s+(?:information|info)\b",
    r"\binsufficient\s+(?:information|info)\b", r"\bcannot\s+determine\b", r"\bcan't\s+determine\b",
    r"\bunable\s+to\s+determine\b", r"\bbased\s+on\s+the\s+provided\s+context\b",
]

def _looks_unknown(raw: str) -> bool:
    t = str(raw or "").strip().lower()
    if not t: return True
    for p in _UNKNOWN_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE): return True
    if len(t) <= 12 and t not in {"none", "true", "false", "yes", "no"}: return True
    return False

def _fallback_gap_for_question(q_type: str) -> str:
    q = str(q_type or "").strip().upper()
    if q == "MC": return "key fact that distinguishes the correct option"
    if q == "TF": return "evidence to decide if the statement is true or false"
    return "key missing clinical detail to answer the question"

def _is_empty(x: str) -> bool: return (x is None) or (not str(x).strip())
def _is_none_strict(x: str) -> bool: return str(x).strip().upper() == "NONE"

def _clean_and_validate_gap(raw_gap: str) -> str:
    gap = str(raw_gap or "").strip().strip("`\"'")
    gap = re.sub(r"\s+", " ", gap)
    if not gap or gap.upper() == "NONE": return "NONE"
    if re.search(r"(?i)\boption\b", gap) or re.fullmatch(r"[A-E]", gap): return "NONE"
    if len(gap) > 120 or len(gap.split()) > 15: return "NONE"
    return gap

def _parse_critic_response(x: str, q_type: str = "QA") -> Dict[str, Any]:
    raw = str(x or "").strip()
    default_gap = _fallback_gap_for_question(q_type)
    parsed: Dict[str, Any] = {"raw": raw, "sufficient": None, "gap": "", "normalized_gap": "", "used_structured": False, "is_unknown": False}
    
    if not raw or _looks_unknown(raw):
        parsed["is_unknown"] = True
        parsed["sufficient"] = False
        parsed["gap"] = default_gap
        parsed["normalized_gap"] = default_gap
        return parsed

    json_str = re.sub(r"```json|```", "", raw).strip()
    try:
        match = re.search(r"\{.*?\}", json_str, re.DOTALL)
        if match: json_str = match.group(0)
        data = json.loads(json_str)
        if "sufficient" in data:
            parsed["sufficient"] = bool(data["sufficient"])
            parsed["gap"] = _clean_and_validate_gap(str(data.get("gap", "NONE")))
            parsed["used_structured"] = True
            parsed["normalized_gap"] = "NONE" if parsed["sufficient"] else (parsed["gap"] if parsed["gap"] != "NONE" else default_gap)
            return parsed
    except Exception: pass

    sufficient_match = re.search(r"(?im)^\s*sufficient\s*:\s*(yes|no|true|false)\b", raw)
    gap_match = re.search(r"(?im)^\s*gap\s*:\s*(.+)$", raw)
    if sufficient_match:
        parsed["used_structured"] = True
        parsed["sufficient"] = sufficient_match.group(1).strip().upper() in {"YES", "TRUE"}
    if gap_match:
        parsed["used_structured"] = True
        parsed["gap"] = _clean_and_validate_gap(gap_match.group(1).strip())

    if parsed["used_structured"]:
        parsed["normalized_gap"] = "NONE" if parsed["sufficient"] else (parsed["gap"] if parsed["gap"] != "NONE" else default_gap)
        return parsed
        
    return parsed

def _normalize_gap_text(x: str, q_type: str = "QA") -> str:
    t = str(x or "").strip().strip("`")
    if not t: return ""
    parsed = _parse_critic_response(x, q_type=q_type)
    return str(parsed["normalized_gap"]) if parsed.get("normalized_gap") else _clean_and_validate_gap(t)

def _is_valid_gap(x: str, q_type: str = "QA") -> bool:
    t = _normalize_gap_text(x, q_type=q_type)
    if not t or t == "NONE" or len(t) < 3 or len(t) > 120 or len(t.split()) > 15: return False
    if t.endswith("...") or "\n" in t: return False
    if re.search(r"(?i)\b(context|question|statement|answer|explanation)\s*:", t): return False
    if re.search(r"(?i)\b(true or false|option\s+[A-E]|assistant|system|user)\b", t): return False
    return True

def _extract_mc_options(question: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for line in str(question or "").splitlines():
        m = re.match(r"\s*([A-E])\.\s*(.+?)\s*$", line)
        if m: options[m.group(1).upper()] = m.group(2).strip()
    return options

def _looks_like_direct_answer(x: str, q_type: str, question: str) -> bool:
    t = _normalize_gap_text(x, q_type=q_type).lower().strip()
    if not t: return False
    q_type_u = str(q_type or "").strip().upper()
    if q_type_u == "TF": return t in {"true", "false", "yes", "no", "answer true", "answer false", "answer yes", "answer no"}
    if q_type_u == "MC":
        if re.fullmatch(r"[a-e]", t): return True
        normalized_options = {v.lower() for v in _extract_mc_options(question).values() if v}
        if t in normalized_options: return True
    return False

def _gap_is_negative(gap: str) -> bool:
    g = (gap or "").strip().lower()
    neg_markers = ["does not state", "doesn't state", "not stated", "not mentioned", "lacks", "lack of", "missing:", "context does not", "not explicitly"]
    return any(m in g for m in neg_markers)

def _call_unified_llm(sys_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 40, model_override: Optional[str] = None) -> str:
    try:
        c = get_llm_client()
        target_model = model_override or os.environ.get("GLOBAL_AGENT_MODEL", "gpt-4o")
        extra_kwargs = {"response_format": {"type": "json_object"}} if os.environ.get("ENABLE_STRUCTURED_OUTPUT", "0") == "1" else {}
        _dbg("llm_call", model=target_model, temperature=temperature, sys_prompt=_clip_text(sys_prompt, 900))
        r = c.chat.completions.create(model=target_model, temperature=temperature, max_tokens=max_tokens, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], **extra_kwargs)
        text = (r.choices[0].message.content or "").strip()
        _dbg("llm_return", raw=_clip_text(text, 1000))
        return text or '{"sufficient": false, "gap": "NONE"}'
    except Exception as e:
        logger.error(f"Critic LLM call failed: {e}")
        return '{"sufficient": false, "gap": "NONE"}'

def _make_sentence_windows(context: str, window_sents: int = 2, max_spans: int = 120) -> List[str]:
    if not context: return []
    try: sents = sent_tokenize(context)
    except Exception: sents = [x.strip() for x in context.split("\n") if x.strip()]
    if not sents: return []
    w = max(1, int(window_sents))
    spans = []
    for i in range(len(sents)):
        chunk = " ".join(sents[i: i + w]).strip()
        if chunk: spans.append(chunk)
        if len(spans) >= max_spans: break
    return spans

def _score_span(retriever, query: str, span: str) -> float:
    try: return float(retriever.score_text(query=query, text=span))
    except TypeError: return float(retriever.score_text(query, span))
    except Exception: return -1.0

def _verify_locate_with_reranker(retriever, gap: str, context: str, window_sents: int = 2, max_spans: int = 120, threshold: float = 0.45) -> Dict[str, Any]:
    spans = _make_sentence_windows(context, window_sents=window_sents, max_spans=max_spans)
    if not spans or retriever is None: 
        return {"present": False, "best_score": None, "best_span": "", "threshold": threshold, "n_spans": len(spans), "window_sents": window_sents}
    
    best_score, best_span = None, ""
    for sp in spans:
        sc = _score_span(retriever, query=gap, span=sp)
        if best_score is None or sc > best_score: best_score, best_span = sc, sp

    preliminary_present = (best_score is not None) and (best_score >= float(threshold))
    final_present = preliminary_present
    target_model = os.environ.get("GLOBAL_AGENT_MODEL", "gpt-4o")

    if preliminary_present and best_span:
        prompt_verify = f"Determine if the following text snippet explicitly provides the missing information described as: '{gap}'.\nIf the text is merely asking a similar question or does not contain the factual answer, reply NO.\nSnippet: {best_span}\nDoes the snippet provide the information? Reply ONLY YES or NO."
        llm_response = _call_unified_llm("You are a strict text verification system.", prompt_verify, temperature=0.0, max_tokens=20, model_override=target_model)
        if "NO" in llm_response.upper():
            logger.info(f"LLM Veto applied for Gap: '{gap[:30]}...' (Score was {best_score:.3f})")
            final_present = False

    return {"present": bool(final_present), "best_score": best_score, "best_span": best_span, "threshold": threshold, "n_spans": len(spans), "window_sents": window_sents, "verify_model": target_model}

def _format_itv_markdown(trace: Dict[str, Any]) -> str:
    lines = ["# Critic ItV Trace\n"]
    for k in ["run_dir", "ts", "item_id", "question_id", "q_type", "calls_per_agent", "none_frac_strict", "empty_frac", "invalid_gap_frac", "unknown_frac", "critic_mode", "consensus_gap", "consensus_gap_is_negative", "verify_mode", "verify_label", "verify_best_score", "verify_threshold", "final_is_sufficient", "final_missing_info"]:
        if k in trace: lines.append(f"- {k}: `{trace.get(k)}`")
    lines.append("\n---\n## Question\n```text\n" + (trace.get("question") or "") + "\n```\n")
    lines.append("## Context (FULL)\n```text\n" + (trace.get("context") or "") + "\n```\n")
    lines.append("\n## Identify Votes\n")
    for v in trace.get("votes", []) or []:
        normalized = (v.get("normalized_text") or "").strip()
        suffix = f" => {normalized}" if normalized and normalized != (v.get("text") or "").strip() else ""
        lines.append(f"- Round {v.get('round')} | {v.get('agent')}: ({v.get('label')}) {v.get('text')}{suffix}")
    lines.append("\n## Consensus details\n```json\n" + str(trace.get("consensus_debug", {})) + "\n```\n")
    if trace.get("verify_mode") == "locate_reranker":
        lines.append("\n## VerifyLocate (gap)\n")
        lines.append(f"- Label: **{trace.get('verify_label','')}**")
        lines.append(f"- Best score: `{trace.get('verify_best_score')}` (threshold `{trace.get('verify_threshold')}`)")
        lines.append("\n```text\n" + (trace.get("verify_best_span") or "") + "\n```\n")
    return "\n".join(lines)


def evaluate_sufficiency(question: str, context: str, q_type: str = "MC", calls_per_agent: int = CRITIC_CALLS_PER_AGENT, question_id: str = None, retriever=None, verify_threshold: float = 0.35, verify_window_sents: int = 2, verify_max_spans: int = 120) -> Tuple[bool, str, Dict[str, Any]]:
    ts = datetime.now().isoformat(timespec="seconds")
    run_dir = get_run_dir()
    item_id = make_item_id(question_id, question)

    trace: Dict[str, Any] = {
        "ts": ts, "run_dir": run_dir, "item_id": item_id, "question_id": question_id or item_id,
        "q_type": (q_type or "QA").strip().upper(), "calls_per_agent": calls_per_agent,
        "question": question, "context": context, "votes": [], "none_frac_strict": None,
        "empty_frac": None, "invalid_gap_frac": None, "valid_gaps": [], "invalid_gaps": [],
        "consensus_gap": "", "consensus_gap_is_negative": False, "verify_mode": "",
        "verify_label": "", "verify_best_score": None, "verify_threshold": None,
        "verify_best_span": "", "verify_n_spans": None, "verify_window_sents": None,
        "final_is_sufficient": None, "final_missing_info": "", "md_path": "",
        "consensus_debug": {}, "critic_mode": "consensus", "unknown_frac": None,
    }

    q_type_u = (q_type or "QA").strip().upper()
    sys_prompt = SYS_ID_TF if q_type_u == "TF" else (SYS_ID_MC if q_type_u == "MC" else SYS_ID_QA)

    votes = []
    empty_count, none_strict_count, invalid_gap_count, unknown_count = 0, 0, 0, 0

    logger.info(f"Evaluating sufficiency for item: {item_id}")
    for i in range(int(calls_per_agent)):
        user_prompt = f"Question/Statement:\n{question}\n\nContext:\n{context}\n\nReturn your JSON now:"
        t0 = time.time()
        text = _call_unified_llm(sys_prompt, user_prompt, temperature=float(CRITIC_LLM_TEMPERATURE), max_tokens=80)
        dt = time.time() - t0

        parsed = _parse_critic_response(text, q_type=q_type_u)
        normalized = parsed.get("normalized_gap") or ""

        if _is_empty(text): empty_count += 1
        if _is_none_strict(normalized): none_strict_count += 1
        if normalized and normalized != "NONE" and (not _is_valid_gap(normalized, q_type=q_type_u) or _looks_like_direct_answer(normalized, q_type_u, question)):
            invalid_gap_count += 1
        if parsed.get("is_unknown"): unknown_count += 1

        label = "SUFFICIENT" if (parsed.get("sufficient") is True) else "INSUFFICIENT"
        votes.append({"round": i + 1, "agent": "critic", "latency_s": dt, "text": (text or "").strip(), "normalized_text": normalized, "label": label})

    if CRITIC_ENABLE_RESCUE and int(calls_per_agent) >= 2:
        frac_unknown = unknown_count / max(1, int(calls_per_agent))
        if frac_unknown >= CRITIC_RESCUE_TRIGGER_FRAC:
            trace["critic_mode"] = "rescue_kv"
            rescue_user = f"Question/Statement:\n{question}\n\nContext:\n{context}\n\nOutput the two lines now."
            t0 = time.time()
            rescue_text = _call_unified_llm(_RESCUE_SYS, rescue_user, temperature=0.0, max_tokens=60)
            dt = time.time() - t0
            rescue_parsed = _parse_critic_response(rescue_text, q_type=q_type_u)
            rescue_norm = rescue_parsed.get("normalized_gap") or ""
            rescue_label = "SUFFICIENT" if (rescue_parsed.get("sufficient") is True) else "INSUFFICIENT"
            votes.append({"round": int(calls_per_agent) + 1, "agent": "critic_rescue", "latency_s": dt, "text": (rescue_text or "").strip(), "normalized_text": rescue_norm, "label": rescue_label})

    trace["votes"] = votes
    trace["empty_frac"] = empty_count / max(1, int(calls_per_agent))
    trace["none_frac_strict"] = none_strict_count / max(1, int(calls_per_agent))
    trace["invalid_gap_frac"] = invalid_gap_count / max(1, int(calls_per_agent))
    trace["unknown_frac"] = unknown_count / max(1, int(calls_per_agent))

    valid_gaps, invalid_gaps = [], []
    for v in votes:
        ng = (v.get("normalized_text") or "").strip()
        if not ng or ng == "NONE": continue
        if _is_valid_gap(ng, q_type=q_type_u) and not _looks_like_direct_answer(ng, q_type_u, question):
            valid_gaps.append(ng)
        else:
            invalid_gaps.append(ng)

    trace["valid_gaps"] = valid_gaps
    trace["invalid_gaps"] = invalid_gaps

    any_sufficient = any(v.get("label") == "SUFFICIENT" for v in votes)

    diverse_gaps = []
    if not valid_gaps:
        if any_sufficient:
            # BUG FIX: If models voted SUFFICIENT, there are no valid gaps, and we shouldn't fallback.
            diverse_gaps = ["NONE"]
            trace["consensus_debug"] = {"status": "sufficient", "selected": ["NONE"]}
            logger.info("Consensus is SUFFICIENT. No valid gaps to verify.")
        else:
            # Models voted INSUFFICIENT but failed to provide valid gap formatting
            diverse_gaps = [_fallback_gap_for_question(q_type_u)]
            trace["consensus_debug"] = {"status": "fallback", "selected": diverse_gaps}
            logger.info(f"No valid gaps extracted. Fallback to default: {diverse_gaps[0]}")
    elif len(valid_gaps) == 1:
        diverse_gaps = valid_gaps
        trace["consensus_debug"] = {"status": "single_gap", "selected": diverse_gaps}
        logger.info(f"Single valid gap extracted: {diverse_gaps[0]}")
    else:
        emb_model = get_emb_model()
        embeddings = emb_model.encode(valid_gaps, convert_to_tensor=True)
        sim_matrix = cos_sim(embeddings, embeddings)
        
        avg_sims = sim_matrix.mean(dim=1).tolist()
        best_idx = avg_sims.index(max(avg_sims))
        primary_gap = valid_gaps[best_idx]
        diverse_gaps.append(primary_gap)
        
        sims_to_primary = sim_matrix[best_idx].tolist()
        min_sim = min(sims_to_primary)
        min_idx = sims_to_primary.index(min_sim)
        
        if min_sim < DIVERSITY_THRESHOLD and valid_gaps[min_idx] != primary_gap:
            diverse_gaps.append(valid_gaps[min_idx])
            
        trace["consensus_debug"] = {
            "status": "clustered", 
            "total_extracted": len(valid_gaps), 
            "selected": diverse_gaps, 
            "min_similarity": round(min_sim, 3)
        }
        logger.info(f"Semantic Clustering Complete. Selected {len(diverse_gaps)} diverse gap(s).")

    consensus_gap_str = " | ".join(diverse_gaps)
    trace["consensus_gap"] = consensus_gap_str
    trace["consensus_gap_is_negative"] = any(_gap_is_negative(g) for g in diverse_gaps)

    if diverse_gaps and diverse_gaps[0] != "NONE" and retriever is not None and context:
        trace["verify_mode"] = "locate_reranker"
        absent_gaps = []
        best_overall_score, best_overall_span = -1.0, ""

        for g in diverse_gaps:
            v_res = _verify_locate_with_reranker(retriever, gap=g, context=context, window_sents=int(verify_window_sents), max_spans=int(verify_max_spans), threshold=float(verify_threshold))
            logger.info(f"Verification | Gap: '{g[:40]}...' | Present: {v_res['present']} | Score: {v_res['best_score']}")
            
            if v_res.get("best_score") is not None and v_res["best_score"] > best_overall_score:
                best_overall_score = v_res["best_score"]
                best_overall_span = v_res["best_span"]
                
            if not v_res["present"]:
                absent_gaps.append(g)
        
        trace["verify_best_score"] = best_overall_score if best_overall_score >= 0 else None
        trace["verify_threshold"] = float(verify_threshold)
        trace["verify_best_span"] = best_overall_span
        trace["verify_window_sents"] = int(verify_window_sents)

        if absent_gaps:
            final_is_sufficient = False
            trace["verify_label"] = "ABSENT"
        else:
            final_is_sufficient = True
            trace["verify_label"] = "PRESENT"
    else:
        trace["verify_mode"] = "none"
        trace["verify_label"] = ""
        final_is_sufficient = bool(any_sufficient)
        absent_gaps = diverse_gaps if not any_sufficient else []

    final_missing_info = "" if final_is_sufficient else (" | ".join(absent_gaps))
    trace["final_is_sufficient"] = final_is_sufficient
    trace["final_missing_info"] = final_missing_info

    try:
        md = _format_itv_markdown(trace)
        md_path = write_text("critic", f"critic_{trace['item_id']}.md", md)
        trace["md_path"] = md_path
    except Exception:
        pass

    try:
        write_jsonl("critic", "critic_traces.jsonl", {"trace": trace})
    except Exception:
        pass

    logger.info(f"Critic Verdict | Sufficient: {final_is_sufficient} | Final Missing Info: {final_missing_info}")
    return bool(final_is_sufficient), final_missing_info, trace