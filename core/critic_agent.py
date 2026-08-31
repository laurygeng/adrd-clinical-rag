#!/usr/bin/env python3
"""
Critic Agent (Sufficiency Gate) — Logged ItV + VerifyLocate (Scheme 3)

Unified System:
- All question types (MC, QA, TF) use the Identify-then-Verify (ItV) gap extraction mechanism.
- Extracts missing information and verifies its presence in the context.
"""

import os
import time
import logging
import traceback
import json
import re
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from core import local_google_shim as genai

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

CRITIC_CALLS_PER_AGENT = int(os.environ.get("CRITIC_CALLS_PER_AGENT", "3"))
CRITIC_OPENAI_TEMPERATURE = float(os.environ.get("CRITIC_OPENAI_TEMPERATURE", "0.3"))
CRITIC_GEMINI_TEMPERATURE = float(os.environ.get("CRITIC_GEMINI_TEMPERATURE", "0.3"))


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
_CRITIC_FORMAT_RULES = (
    "You MUST output your answer strictly as a valid JSON object with NO extra text, comments, or markdown.\n"
    "{\n"
    '  "sufficient": true,\n'
    '  "gap": "NONE"\n'
    "}\n"
    "OR\n"
    "{\n"
    '  "sufficient": false,\n'
    '  "gap": "<short missing fact phrase>"\n'
    "}\n"
    "CRITICAL RULES:\n"
    "1. Do NOT solve the question. Do NOT mention any option letters (A, B, C, D, E).\n"
    "2. The 'gap' must be a short medical fact phrase (at most 10 words), NEVER an option name like 'Option D'.\n"
    "3. If the exact direct fact is missing from the context, set sufficient to false."
)

SYS_ID_MC = (
    "You judge whether a CONTEXT is sufficient to answer a multiple-choice question. Identify only whether the "
    "context is sufficient and, if not, the single most important missing fact needed to determine the correct "
    f"option. {_CRITIC_FORMAT_RULES}"
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


# ----------------------------
# Helpers & Guardrails
# ----------------------------
def _is_empty(x: str) -> bool:
    return (x is None) or (not str(x).strip())


def _is_none_strict(x: str) -> bool:
    return str(x).strip().upper() == "NONE"


def _clean_and_validate_gap(raw_gap: str) -> str:
    """[GUARDRAIL]: 清洗与校验小模型产生的 gap，防止 Option D 或长篇废话污染检索"""
    gap = str(raw_gap or "").strip().strip("`\"'")
    gap = re.sub(r"\s+", " ", gap)
    
    if not gap or gap.upper() == "NONE":
        return "NONE"
        
    # 1. 如果包含选项字眼或单独的选项字母（如 Option D, A, B 等），直接判定为无效
    if re.search(r"(?i)\boption\b", gap) or re.fullmatch(r"[A-E]", gap):
        return "NONE"
        
    # 2. 如果包含大段解释性废话或超长文本，截断或降级
    if len(gap) > 100 or len(gap.split()) > 12:
        return "NONE"
        
    return gap


def _parse_critic_response(x: str) -> Dict[str, Any]:
    raw = str(x or "").strip()
    parsed: Dict[str, Any] = {
        "raw": raw,
        "sufficient": None,
        "gap": "",
        "normalized_gap": "",
        "used_structured": False,
    }
    if not raw:
        return parsed

    # 1. Try to parse as JSON first (Best for Llama-3/Med-7B)
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
            
            if parsed["sufficient"] is True or parsed["gap"] == "NONE":
                parsed["normalized_gap"] = "NONE"
            else:
                parsed["normalized_gap"] = parsed["gap"]
            return parsed
    except Exception:
        pass 

    # 2. Fallback to regex extraction
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
        if parsed["sufficient"] is True or parsed["gap"] == "NONE":
            parsed["normalized_gap"] = "NONE"
        else:
            parsed["normalized_gap"] = parsed["gap"]
        return parsed

    upper = raw.upper()
    if upper == "NONE":
        parsed["normalized_gap"] = "NONE"
        return parsed

    return parsed


def _normalize_gap_text(x: str) -> str:
    t = str(x or "").strip().strip("`")
    if not t:
        return ""

    parsed = _parse_critic_response(x)
    if parsed.get("normalized_gap"):
        return str(parsed["normalized_gap"])

    cleaned = _clean_and_validate_gap(t)
    return cleaned


def _is_valid_gap(x: str) -> bool:
    t = _normalize_gap_text(x)
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
    t = _normalize_gap_text(x)
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


# ----------------------------
# Model calls: Identify
# ----------------------------
def _call_openai(sys_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 40) -> str:
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
            response_format={"type": "json_object"}
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        logging.warning(f"Critic openai call failed: {e}")
        return ""


def _call_gemini(sys_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 120) -> str:
    try:
        client = get_gem_client()
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[f"SYSTEM:\n{sys_prompt}\n\nUSER:\n{user_prompt}\n"],
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        txt = getattr(resp, "text", "") or ""
        return str(txt).strip()
    except Exception as e:
        logging.warning(f"Critic gemini call failed: {e}")
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


def _verify_locate_with_reranker(
    retriever,
    gap: str,
    context: str,
    window_sents: int = 2,
    max_spans: int = 120,
    threshold: float = 0.45,
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


# ----------------------------
# Main entry
# ----------------------------
def evaluate_sufficiency(
    question: str,
    context: str,
    q_type: str = "MC",
    calls_per_agent: int = CRITIC_CALLS_PER_AGENT,
    question_id: str = None,
    retriever=None,
    verify_threshold: float = 0.35,
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

        "verify_mode": "",
        "verify_label": "SKIPPED",
        "verify_best_score": None,
        "verify_best_span": "",
        "verify_threshold": verify_threshold,
        "verify_n_spans": None,
        "verify_window_sents": verify_window_sents,

        "final_is_sufficient": True,
        "final_missing_info": "",
        "md_path": "",
    }

    q_type_u = trace["q_type"]

    try:
        if retriever is None:
            raise ValueError("evaluate_sufficiency requires retriever for VerifyLocate scoring.")

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
        calls_per_agent = max(1, int(calls_per_agent or CRITIC_CALLS_PER_AGENT))
        trace["calls_per_agent"] = calls_per_agent

        for r in range(1, calls_per_agent + 1):
            o = _call_openai(
                sys_id,
                user_prompt,
                temperature=CRITIC_OPENAI_TEMPERATURE,
                max_tokens=50,
            )
            g = _call_gemini(
                sys_id,
                user_prompt,
                temperature=CRITIC_GEMINI_TEMPERATURE,
                max_tokens=60,
            )

            for agent, txt in (("openai", o), ("gemini", g)):
                n_votes += 1
                parsed = _parse_critic_response(txt)
                normalized = _normalize_gap_text(txt)
                direct_answer = _looks_like_direct_answer(normalized or txt, q_type_u, question)
                
                if _is_empty(txt):
                    label = "EMPTY"
                    empty_count += 1
                elif parsed.get("sufficient") is True or _is_none_strict(normalized or txt) or normalized == "NONE":
                    label = "NONE"
                    none_count_strict += 1
                else:
                    if direct_answer:
                        label = "GAP_INVALID"
                        invalid_gaps.append((normalized or txt).strip())
                    elif _is_valid_gap(normalized):
                        label = "GAP_VALID"
                        valid_gaps.append(normalized.strip())
                    else:
                        label = "GAP_INVALID"
                        invalid_gaps.append((normalized or txt).strip())

                trace["votes"].append({
                    "round": r,
                    "agent": agent,
                    "text": txt,
                    "parsed_sufficient": parsed.get("sufficient"),
                    "used_structured": parsed.get("used_structured", False),
                    "normalized_text": normalized,
                    "label": label,
                })

        trace["none_frac_strict"] = float(none_count_strict / max(n_votes, 1))
        trace["empty_frac"] = float(empty_count / max(n_votes, 1))
        trace["invalid_gap_frac"] = float(len(invalid_gaps) / max(n_votes, 1))
        trace["valid_gaps"] = valid_gaps
        trace["invalid_gaps"] = invalid_gaps

        if trace["none_frac_strict"] >= 0.6:
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
        trace["error"] = str(e)
        trace["traceback"] = traceback.format_exc()

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