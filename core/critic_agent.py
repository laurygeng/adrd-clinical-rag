#!/usr/bin/env python3
"""
Critic Agent (Sufficiency Gate) — conservative logic + detailed debug logging.
"""

import os
import time
import logging
import json
import re
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

CRITIC_CALLS_PER_AGENT = int(os.environ.get("CRITIC_CALLS_PER_AGENT", "3"))
CRITIC_LLM_TEMPERATURE = float(os.environ.get("CRITIC_OPENAI_TEMPERATURE", "0.3"))
CRITIC_ENABLE_RESCUE = os.environ.get("CRITIC_ENABLE_RESCUE", "1") == "1"
CRITIC_RESCUE_TRIGGER_FRAC = float(os.environ.get("CRITIC_RESCUE_TRIGGER_FRAC", "0.67"))


def get_emb_model() -> SentenceTransformer:
    global _emb_model
    if _emb_model is None:
        logging.info("Loading local embedding model for consensus voting...")
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

SYS_ID_MC = (
    "You judge whether a CONTEXT is sufficient to answer a multiple-choice question. Identify only whether the "
    "context is sufficient and, if not, the single most important missing fact needed to determine the correct option. "
    f"{_CRITIC_FORMAT_RULES}"
)
SYS_ID_TF = (
    "You judge whether a CONTEXT is sufficient to decide if a TRUE/FALSE statement is true or false. Identify only "
    "whether the context is sufficient and, if not, the single most important missing fact needed to decide. "
    f"{_CRITIC_FORMAT_RULES}"
)
SYS_ID_QA = (
    "You judge whether a CONTEXT is sufficient to answer an open-ended question completely and accurately. Identify "
    "only whether the context is sufficient and, if not, the single most important missing fact needed to answer. "
    f"{_CRITIC_FORMAT_RULES}"
)

_RESCUE_SYS = (
    "You are a sufficiency judge. Output EXACTLY two lines and nothing else:\n"
    "sufficient: yes|no\n"
    "gap: NONE|<short missing fact phrase>\n"
    "If you are unsure, choose sufficient: no and provide the single most important missing fact phrase.\n"
    "NEVER say 'I don't know'."
)

_UNKNOWN_PATTERNS = [
    r"\bi\s*don't\s*know\b",
    r"\bi\s*do\s*not\s*know\b",
    r"\bnot\s+enough\s+(?:information|info)\b",
    r"\binsufficient\s+(?:information|info)\b",
    r"\bcannot\s+determine\b",
    r"\bcan't\s+determine\b",
    r"\bunable\s+to\s+determine\b",
    r"\bbased\s+on\s+the\s+provided\s+context\b",
    r"\b不知道\b",
    r"\b信息不足\b",
    r"\b无法判断\b",
    r"\b无法确定\b",
]


def _looks_unknown(raw: str) -> bool:
    t = str(raw or "").strip().lower()
    if not t:
        return True
    for p in _UNKNOWN_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE):
            return True
    if len(t) <= 12 and t not in {"none", "true", "false", "yes", "no"}:
        return True
    return False


def _fallback_gap_for_question(q_type: str) -> str:
    q = str(q_type or "").strip().upper()
    if q == "MC":
        return "key fact that distinguishes the correct option"
    if q == "TF":
        return "evidence to decide if the statement is true or false"
    return "key missing clinical detail to answer the question"


def _is_empty(x: str) -> bool:
    return (x is None) or (not str(x).strip())


def _is_none_strict(x: str) -> bool:
    return str(x).strip().upper() == "NONE"


def _clean_and_validate_gap(raw_gap: str) -> str:
    gap = str(raw_gap or "").strip().strip("`\"'")
    gap = re.sub(r"\s+", " ", gap)
    if not gap or gap.upper() == "NONE":
        return "NONE"
    if re.search(r"(?i)\boption\b", gap) or re.fullmatch(r"[A-E]", gap):
        return "NONE"
    if len(gap) > 100 or len(gap.split()) > 12:
        return "NONE"
    return gap


def _parse_critic_response(x: str, q_type: str = "QA") -> Dict[str, Any]:
    raw = str(x or "").strip()
    _dbg("parse_begin", q_type=q_type, raw=_clip_text(raw, 1200))
    default_gap = _fallback_gap_for_question(q_type)
    parsed: Dict[str, Any] = {
        "raw": raw,
        "sufficient": None,
        "gap": "",
        "normalized_gap": "",
        "used_structured": False,
        "is_unknown": False,
    }
    if not raw:
        parsed["is_unknown"] = True
        _dbg("parse_empty", parsed=parsed)
        return parsed

    if _looks_unknown(raw):
        parsed["is_unknown"] = True
        parsed["sufficient"] = False
        parsed["gap"] = default_gap
        parsed["normalized_gap"] = default_gap
        _dbg("parse_unknown", default_gap=default_gap, parsed=parsed)
        return parsed

    json_str = re.sub(r"```json|```", "", raw).strip()
    try:
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            json_str = match.group(0)
        data = json.loads(json_str)
        if "sufficient" in data:
            parsed["sufficient"] = bool(data["sufficient"])
            raw_gap = str(data.get("gap", "NONE"))
            parsed["gap"] = _clean_and_validate_gap(raw_gap)
            parsed["used_structured"] = True
            if parsed["sufficient"] is True:
                parsed["normalized_gap"] = "NONE"
            else:
                parsed["normalized_gap"] = parsed["gap"] if parsed["gap"] != "NONE" else default_gap
            _dbg("parse_json_ok", parsed=parsed)
            return parsed
    except Exception as e:
        _dbg("parse_json_failed", error=str(e), raw=_clip_text(raw, 800))

    sufficient_match = re.search(r"(?im)^\s*sufficient\s*:\s*(yes|no|true|false)\b", raw)
    gap_match = re.search(r"(?im)^\s*gap\s*:\s*(.+)$", raw)
    if sufficient_match:
        parsed["used_structured"] = True
        sufficient_token = sufficient_match.group(1).strip().upper()
        parsed["sufficient"] = sufficient_token in {"YES", "TRUE"}
    if gap_match:
        parsed["used_structured"] = True
        parsed["gap"] = _clean_and_validate_gap(gap_match.group(1).strip())

    if parsed["used_structured"]:
        if parsed["sufficient"] is True:
            parsed["normalized_gap"] = "NONE"
        else:
            parsed["normalized_gap"] = parsed["gap"] if parsed["gap"] != "NONE" else default_gap
        _dbg("parse_kv_ok", parsed=parsed)
        return parsed

    lower_raw = raw.lower()
    if re.search(r"\b(not sufficient|insufficient)\b", lower_raw):
        parsed["sufficient"] = False
        parsed["used_structured"] = True
    elif re.search(r"\bsufficient\b", lower_raw):
        parsed["sufficient"] = True
        parsed["used_structured"] = True

    freeform_gap_patterns = [
        r"(?i)missing (?:fact|information|info)(?: needed)?(?: is|:)?\s*(.+)$",
        r"(?i)gap(?: is|:)?\s*(.+)$",
        r"(?i)needed (?:fact|information|info)(?: is|:)?\s*(.+)$",
        r"(?i)the context (?:is )?missing\s+(.+)$",
    ]
    for pattern in freeform_gap_patterns:
        match = re.search(pattern, raw)
        if not match:
            continue
        candidate_gap = _clean_and_validate_gap(match.group(1).strip())
        if candidate_gap and candidate_gap != "NONE":
            parsed["gap"] = candidate_gap
            parsed["used_structured"] = True
            break

    if parsed["used_structured"]:
        if parsed["sufficient"] is True:
            parsed["normalized_gap"] = "NONE"
        else:
            parsed["normalized_gap"] = parsed["gap"] if parsed["gap"] != "NONE" else default_gap
        _dbg("parse_freeform_ok", parsed=parsed)
        return parsed

    _dbg("parse_unstructured_fallback", parsed=parsed)
    return parsed


def _normalize_gap_text(x: str, q_type: str = "QA") -> str:
    t = str(x or "").strip().strip("`")
    if not t:
        return ""
    parsed = _parse_critic_response(x, q_type=q_type)
    if parsed.get("normalized_gap"):
        return str(parsed["normalized_gap"])
    return _clean_and_validate_gap(t)


def _is_valid_gap(x: str, q_type: str = "QA") -> bool:
    t = _normalize_gap_text(x, q_type=q_type)
    if not t or t == "NONE":
        return False
    if len(t) < 3 or len(t) > 120:
        return False
    if len(t.split()) > 12:
        return False
    if t.endswith("..."):
        return False
    if "\n" in t:
        return False
    if re.search(r"(?i)\b(context|question|statement|answer|explanation)\s*:", t):
        return False
    if re.search(r"(?i)\b(true or false|option\s+[A-E]|assistant|system|user)\b", t):
        return False
    return True


def _extract_mc_options(question: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for line in str(question or "").splitlines():
        m = re.match(r"\s*([A-E])\.\s*(.+?)\s*$", line)
        if m:
            options[m.group(1).upper()] = m.group(2).strip()
    return options


def _looks_like_direct_answer(x: str, q_type: str, question: str) -> bool:
    t = _normalize_gap_text(x, q_type=q_type)
    if not t:
        return False
    q_type_u = str(q_type or "").strip().upper()
    lower_t = t.lower().strip()
    if q_type_u == "TF":
        return lower_t in {"true", "false", "yes", "no", "answer true", "answer false", "answer yes", "answer no"}
    if q_type_u == "MC":
        if re.fullmatch(r"[A-E]", t):
            return True
        option_match = re.match(r"^([A-E])(?:[\).:\-]|\s)+(.*)$", t)
        options = _extract_mc_options(question)
        if option_match:
            letter = option_match.group(1).upper()
            remainder = option_match.group(2).strip().lower()
            if letter in options:
                option_text = options[letter].lower()
                if not remainder or remainder == option_text or option_text in remainder or remainder in option_text:
                    return True
        normalized_options = {v.lower() for v in _extract_mc_options(question).values() if v}
        if lower_t in normalized_options:
            return True
    return False


def _gap_is_negative(gap: str) -> bool:
    g = (gap or "").strip().lower()
    neg_markers = [
        "does not state", "doesn't state", "not stated", "not mentioned",
        "not explicitly", "isn't mentioned", "is not mentioned", "lacks",
        "lack of", "missing:", "context does not", "the context does not",
    ]
    return any(m in g for m in neg_markers)


_SAFE_JSON_FALLBACK = '{"sufficient": false, "gap": "key information to answer the question"}'


def _call_unified_llm(sys_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 40, model_override: Optional[str] = None) -> str:
    try:
        c = get_llm_client()
        target_model = model_override or os.environ.get("GLOBAL_AGENT_MODEL", "gpt-4o")
        extra_kwargs: Dict[str, Any] = {}
        if os.environ.get("ENABLE_STRUCTURED_OUTPUT", "0") == "1":
            extra_kwargs["response_format"] = {"type": "json_object"}

        _dbg("llm_call", model=target_model, temperature=temperature, max_tokens=max_tokens, sys_prompt=_clip_text(sys_prompt, 900), user_prompt=_clip_text(user_prompt, 1200))
        r = c.chat.completions.create(
            model=target_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **extra_kwargs,
        )
        text = (r.choices[0].message.content or "").strip()
        _dbg("llm_return", raw=_clip_text(text, 1000))
        return text or _SAFE_JSON_FALLBACK
    except Exception as e:
        logging.warning(f"Critic LLM call failed: {e}")
        _dbg("llm_error", error=str(e))
        return _SAFE_JSON_FALLBACK


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


def _verify_locate_with_reranker(retriever, gap: str, context: str, window_sents: int = 2, max_spans: int = 120, threshold: float = 0.45) -> Dict[str, Any]:
    spans = _make_sentence_windows(context, window_sents=window_sents, max_spans=max_spans)
    _dbg("verify_begin", gap=gap, context_chars=len(context or ""), n_spans=len(spans), threshold=threshold)
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

    preliminary_present = (best_score is not None) and (best_score >= float(threshold))
    final_present = preliminary_present

    target_model = os.environ.get("GLOBAL_AGENT_MODEL", "gpt-4o")
    if preliminary_present and best_span:
        prompt_verify = (
            f"Determine if the following text snippet explicitly provides the missing information described as: '{gap}'.\n"
            f"If the text is merely asking a similar question, acting as a section header, or does not contain the actual factual answer, reply NO.\n"
            f"Snippet: {best_span}\n"
            f"Does the snippet provide the information? Reply ONLY YES or NO."
        )
        llm_response = _call_unified_llm(
            "You are a strict text verification system.",
            prompt_verify,
            temperature=0.0,
            max_tokens=50,
            model_override=target_model,
        )
        if "NO" in llm_response.upper():
            logging.info(f"LLM Veto applied: Reranker score was {best_score:.3f} but LLM rejected the snippet (model={target_model}).")
            final_present = False

    result = {
        "present": bool(final_present),
        "best_score": best_score,
        "best_span": best_span,
        "threshold": threshold,
        "n_spans": len(spans),
        "window_sents": window_sents,
        "verify_model": target_model,
    }
    _dbg("verify_end", result=result)
    return result


def _format_itv_markdown(trace: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Critic ItV Trace\n")
    for k in [
        "run_dir", "ts", "item_id", "question_id", "q_type", "calls_per_agent",
        "none_frac_strict", "empty_frac", "invalid_gap_frac", "unknown_frac", "critic_mode",
        "consensus_gap", "consensus_gap_is_negative",
        "verify_mode", "verify_label", "verify_best_score", "verify_threshold",
        "final_is_sufficient", "final_missing_info"
    ]:
        if k in trace:
            lines.append(f"- {k}: `{trace.get(k)}`")
    lines.append("\n---\n")
    lines.append("## Question\n```text\n" + (trace.get("question") or "") + "\n```\n")
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
        "verify_mode": "",
        "verify_label": "",
        "verify_best_score": None,
        "verify_threshold": None,
        "verify_best_span": "",
        "verify_n_spans": None,
        "verify_window_sents": None,
        "final_is_sufficient": None,
        "final_missing_info": "",
        "md_path": "",
        "consensus_debug": {},
        "critic_mode": "consensus",
        "unknown_frac": None,
    }

    q_type_u = (q_type or "QA").strip().upper()
    if q_type_u == "TF":
        sys_prompt = SYS_ID_TF
    elif q_type_u == "MC":
        sys_prompt = SYS_ID_MC
    else:
        sys_prompt = SYS_ID_QA

    votes = []
    empty_count = 0
    none_strict_count = 0
    invalid_gap_count = 0
    unknown_count = 0

    for i in range(int(calls_per_agent)):
        user_prompt = (
            f"Question/Statement:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Return your JSON now:"
        )

        _dbg("identify_prompt", round=i + 1, q_type=q_type_u, system=_clip_text(sys_prompt, 900), user=_clip_text(user_prompt, 1200), context_chars=len(context or ""), question_chars=len(question or ""))
        t0 = time.time()
        text = _call_unified_llm(
            sys_prompt,
            user_prompt,
            temperature=float(CRITIC_LLM_TEMPERATURE),
            max_tokens=80,
        )
        dt = time.time() - t0

        parsed = _parse_critic_response(text, q_type=q_type_u)
        normalized = parsed.get("normalized_gap") or ""

        if _is_empty(text):
            empty_count += 1
        if _is_none_strict(normalized):
            none_strict_count += 1
        if normalized and normalized != "NONE" and (not _is_valid_gap(normalized, q_type=q_type_u) or _looks_like_direct_answer(normalized, q_type_u, question)):
            invalid_gap_count += 1
        if parsed.get("is_unknown"):
            unknown_count += 1

        suff = parsed.get("sufficient")
        label = "SUFFICIENT" if (suff is True) else "INSUFFICIENT"
        vote_obj = {
            "round": i + 1,
            "agent": "critic",
            "latency_s": dt,
            "text": (text or "").strip(),
            "normalized_text": normalized,
            "label": label,
        }
        votes.append(vote_obj)
        _dbg("identify_vote", round=i + 1, latency_s=dt, label=label, raw=_clip_text(text, 800), normalized=normalized, parsed=parsed)

    if CRITIC_ENABLE_RESCUE and int(calls_per_agent) >= 2:
        frac_unknown = unknown_count / max(1, int(calls_per_agent))
        if frac_unknown >= CRITIC_RESCUE_TRIGGER_FRAC:
            trace["critic_mode"] = "rescue_kv"
            rescue_user = (
                f"Question/Statement:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Output the two lines now."
            )
            _dbg("rescue_prompt", system=_clip_text(_RESCUE_SYS, 900), user=_clip_text(rescue_user, 1200), frac_unknown=frac_unknown)
            t0 = time.time()
            rescue_text = _call_unified_llm(
                _RESCUE_SYS,
                rescue_user,
                temperature=0.0,
                max_tokens=60,
            )
            dt = time.time() - t0
            rescue_parsed = _parse_critic_response(rescue_text, q_type=q_type_u)
            rescue_norm = rescue_parsed.get("normalized_gap") or ""
            rescue_suff = rescue_parsed.get("sufficient")
            rescue_label = "SUFFICIENT" if (rescue_suff is True) else "INSUFFICIENT"
            rescue_vote = {
                "round": int(calls_per_agent) + 1,
                "agent": "critic_rescue",
                "latency_s": dt,
                "text": (rescue_text or "").strip(),
                "normalized_text": rescue_norm,
                "label": rescue_label,
            }
            votes.append(rescue_vote)
            _dbg("rescue_vote", latency_s=dt, label=rescue_label, raw=_clip_text(rescue_text, 800), normalized=rescue_norm, parsed=rescue_parsed)

    trace["votes"] = votes
    trace["empty_frac"] = empty_count / max(1, int(calls_per_agent))
    trace["none_frac_strict"] = none_strict_count / max(1, int(calls_per_agent))
    trace["invalid_gap_frac"] = invalid_gap_count / max(1, int(calls_per_agent))
    trace["unknown_frac"] = unknown_count / max(1, int(calls_per_agent))

    valid_gaps = []
    invalid_gaps = []
    for v in votes:
        ng = (v.get("normalized_text") or "").strip()
        if not ng or ng == "NONE":
            continue
        if _is_valid_gap(ng, q_type=q_type_u) and not _looks_like_direct_answer(ng, q_type_u, question):
            valid_gaps.append(ng)
        else:
            invalid_gaps.append(ng)

    trace["valid_gaps"] = valid_gaps
    trace["invalid_gaps"] = invalid_gaps

    default_gap = _fallback_gap_for_question(q_type_u)
    consensus_gap = valid_gaps[0] if valid_gaps else default_gap
    trace["consensus_gap"] = consensus_gap
    trace["consensus_gap_is_negative"] = _gap_is_negative(consensus_gap)
    any_sufficient = any(v.get("label") == "SUFFICIENT" for v in votes)
    _dbg("consensus_pre_verify", valid_gaps=valid_gaps, invalid_gaps=invalid_gaps, default_gap=default_gap, consensus_gap=consensus_gap, any_sufficient=any_sufficient, unknown_frac=trace["unknown_frac"])

    verify = None
    if consensus_gap != "NONE" and retriever is not None and context:
        trace["verify_mode"] = "locate_reranker"
        verify = _verify_locate_with_reranker(
            retriever=retriever,
            gap=consensus_gap,
            context=context,
            window_sents=int(verify_window_sents),
            max_spans=int(verify_max_spans),
            threshold=float(verify_threshold),
        )
        present = bool(verify.get("present"))
        trace["verify_best_score"] = verify.get("best_score")
        trace["verify_threshold"] = verify.get("threshold")
        trace["verify_best_span"] = verify.get("best_span")
        trace["verify_n_spans"] = verify.get("n_spans")
        trace["verify_window_sents"] = verify.get("window_sents")
        trace["verify_label"] = "PRESENT" if present else "ABSENT"
        final_is_sufficient = bool(present) or bool(any_sufficient)
        _dbg("verify_result", verify=verify, final_is_sufficient=final_is_sufficient)
    else:
        trace["verify_mode"] = "none"
        trace["verify_label"] = ""
        final_is_sufficient = bool(any_sufficient)
        _dbg("verify_skipped", reason="no_consensus_gap_or_no_retriever_or_no_context", final_is_sufficient=final_is_sufficient)

    missing_info = "" if final_is_sufficient else consensus_gap
    trace["final_is_sufficient"] = bool(final_is_sufficient)
    trace["final_missing_info"] = (missing_info or "").strip()

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

    _dbg("evaluate_end", final_is_sufficient=final_is_sufficient, missing_info=missing_info, verify_label=trace.get("verify_label"), md_path=trace.get("md_path"))
    return bool(final_is_sufficient), (missing_info or "").strip(), trace
