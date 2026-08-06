#!/usr/bin/env python3
"""
Orchestrator (TF Iter-3) — Hardened

Key hardening:
- TF NLI label must ALWAYS exist (never empty/NaN). If missing, emergency rerun on available context.
- Normalize legacy/variant labels (e.g., CONTRADICTED -> HARD_CONTRADICTION).
- Injection is audit-only and MUST NOT gate NLI.
- TF final answer still calls LLM (business requirement) but is locked to deterministic mapping.
"""

import os
import logging
import hashlib
from typing import Dict, Any, List, Union, Optional
from datetime import datetime

from openai import OpenAI

from core.advanced_retriever import AdvancedRetriever
from core.critic_agent import evaluate_sufficiency, rerun_tf_nli
from core.answer_agent import generate_final_answer, generate_tf_final_answer_locked
from core.search_agent import research
from core.trace_logger import write_jsonl, make_item_id, write_run_meta, get_run_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_retriever_instance = None
_oai_client = None

TF_LABELS = {"SUPPORTED", "HARD_CONTRADICTION", "SOFT_CONTRADICTION", "NOT_ENOUGH_INFO"}


def _normalize_tf_label(label: str) -> str:
    lab = (label or "").strip().upper()
    if lab in ("CONTRADICTED", "HARD_CONTRADICTION"):
        return "HARD_CONTRADICTION"
    if lab in ("SOFT_CONTRADICTION",):
        return "SOFT_CONTRADICTION"
    if lab in ("SUPPORTED", "NOT_ENOUGH_INFO"):
        return lab
    return ""


def _ensure_tf_label(trace_log: Dict[str, Any], field: str, default: str = "NOT_ENOUGH_INFO") -> str:
    """
    Ensure a TF NLI label field exists and is valid.
    Returns normalized label (guaranteed non-empty and within TF_LABELS).
    """
    lab = _normalize_tf_label(trace_log.get(field, ""))
    if lab in TF_LABELS:
        trace_log[field] = lab
        return lab
    trace_log[field] = default
    return default


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
    # keep it short-ish
    return g[:800]


def build_tf_completion_query(statement: str, consensus_gap: str) -> str:
    gap = sanitize_gap_text(consensus_gap)
    if gap:
        return f"{statement.strip()}\n\nMissing info to find (hint): {gap}".strip()
    return statement.strip()


# def generate_web_rewrite_query(statement: str, gap_hint: str = "") -> str:
#     s = statement.strip()
#     gh = sanitize_gap_text(gap_hint)
#     if gh:
#         return f"{s}\n\nSearch query focus: {gh}".strip()
#     return s
def generate_web_rewrite_query(statement: str, gap_hint: str = "") -> str:
    """
    [方案3 优化版] 使用 LLM 深度提取和改写定向突破的 Google 搜索词。
    将原陈述与缺失信息(Gap)结合，生成高精度搜索词，避免泛泛而谈。
    """
    s = statement.strip()
    gh = sanitize_gap_text(gap_hint)
    
    if not gh:
        return s
        
    client = get_oai_client()
    sys_prompt = "You are an expert medical search query generator."
    
    # 引导 LLM 生成专门用于“验证缺失细节”的高精度检索词
    user_prompt = (
        f"We are trying to verify this True/False statement:\n\"{s}\"\n\n"
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
        # 降级回退到原有的简单拼接
        return f"{s}\n\nSearch query focus: {gh}".strip()

def _pick_span_for_injection(trace_log: Dict[str, Any], use_hop2: bool = False) -> str:
    spans_key = "tf_support_top_spans_hop2" if use_hop2 else "tf_support_top_spans"
    spans = trace_log.get(spans_key, []) or []
    if spans:
        return str(spans[0])
    # fallback to any best span fields if present
    return ""


def build_tf_top_evidence_block(span: str, nli_label: str, citations: List[str]) -> str:
    span = (span or "").strip()
    if not span:
        return ""
    cits = [str(c).strip() for c in (citations or []) if str(c).strip()]
    cits = cits[:5]
    cit_block = "\n".join([f"- {c}" for c in cits]) if cits else ""
    block = [
        "=== TOP EVIDENCE (TF) ===",
        f"NLI_LABEL: {nli_label}",
        "SPAN:",
        span,
    ]
    if cit_block:
        block += ["CITATIONS:", cit_block]
    block.append("=== END TOP EVIDENCE ===")
    return "\n".join(block) + "\n"


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

        # TF pre
        "tf_support_top_spans": [],
        "tf_support_top_scores": [],
        "tf_nli_label": "",
        "tf_nli_confidence": None,
        "tf_nli_best_span_index": None,
        "tf_nli_citations": [],
        "tf_nli_explanation": "",
        "tf_support_injected": False,

        # TF hop1
        "tf_nli_label_post_completion": "",
        "tf_nli_confidence_post_completion": None,
        "tf_nli_best_span_index_post_completion": None,
        "tf_nli_citations_post_completion": [],
        "tf_nli_explanation_post_completion": "",

        # TF hop2
        "tf_nli_label_hop2": "",
        "tf_nli_confidence_hop2": None,
        "tf_nli_best_span_index_hop2": None,
        "tf_nli_citations_hop2": [],
        "tf_nli_explanation_hop2": "",
        "tf_support_top_spans_hop2": [],
        "tf_support_top_scores_hop2": [],

        # triggers
        "completion_triggered": False,
        "completion_reason": "",
        "completion_skipped_reason": "",
        "hop2_triggered": False,
        "hop2_reason": "",

        # final audit
        "tf_final_nli_label_used": "",
        "tf_final_answer_source": "",
        "tf_answer_llm_locked": False,
        "web_query_used": "",
        "web_query_used_hop2": "",
        "court_statement_used": "",
        "court_verdict": "",
        "court_flags": {},
        "veto_triggered": False,
    }

    # STEP 1: Base Retrieval
    logging.info("Step 1: Running Base Retrieval...")
    base_passages, _, _ = retriever.get_retrieved_passages(
        question,
        top_k=8,
        bm25_weight=0.3,
        vector_weight=0.7,
    )
    base_context = "\n\n".join(base_passages)

    # STEP 2: Critic
    logging.info("Step 2: Critic Agent evaluating sufficiency...")
    is_sufficient, missing_info, critic_trace = evaluate_sufficiency(
        question=question,
        context=base_context,
        q_type=q_type,
        calls_per_agent=5,
        question_id=question_id,
        retriever=retriever,
    )

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

        # TF (pre)
        trace_log["tf_support_top_spans"] = critic_trace.get("tf_support_top_spans", []) or []
        trace_log["tf_support_top_scores"] = critic_trace.get("tf_support_top_scores", []) or []
        trace_log["tf_nli_label"] = (critic_trace.get("tf_nli_label") or "").strip()
        trace_log["tf_nli_confidence"] = critic_trace.get("tf_nli_confidence")
        trace_log["tf_nli_best_span_index"] = critic_trace.get("tf_nli_best_span_index")
        trace_log["tf_nli_citations"] = critic_trace.get("tf_nli_citations", []) or []
        trace_log["tf_nli_explanation"] = critic_trace.get("tf_nli_explanation", "") or ""

    q_type_u = (q_type or "").strip().upper()

    # STEP 3-5
    if q_type_u == "TF":
        # --- Emergency: ensure PRE NLI label exists ---
        pre_label = _normalize_tf_label(trace_log.get("tf_nli_label", ""))
        if pre_label not in TF_LABELS:
            logging.warning(f"[TF_PRE_LABEL_MISSING] qid={question_id} pre_label='{trace_log.get('tf_nli_label')}'. Emergency rerun on base_context.")
            try:
                post = rerun_tf_nli(
                    statement=question,
                    context=base_context,
                    retriever=retriever,
                    window_sents=2,
                    max_spans=120,
                    top_k=5,
                )
                nli = (post.get("nli") or {})
                trace_log["tf_nli_label"] = (nli.get("label") or "").strip()
                trace_log["tf_nli_confidence"] = nli.get("confidence")
                trace_log["tf_nli_best_span_index"] = nli.get("best_span_index")
                trace_log["tf_nli_citations"] = nli.get("citations", []) or []
                trace_log["tf_nli_explanation"] = nli.get("explanation", "") or ""
                trace_log["tf_support_top_spans"] = post.get("top_spans", []) or trace_log.get("tf_support_top_spans", [])
                trace_log["tf_support_top_scores"] = post.get("top_scores", []) or trace_log.get("tf_support_top_scores", [])
            except Exception as e:
                logging.error(f"[TF_PRE_LABEL_RERUN_FAILED] qid={question_id}: {e}")

        # finalize/normalize pre label
        nli_label_pre = _ensure_tf_label(trace_log, "tf_nli_label", default="NOT_ENOUGH_INFO")

        # gate: NEI or SOFT => completion
        should_complete = nli_label_pre in ("NOT_ENOUGH_INFO", "SOFT_CONTRADICTION")
        logging.info(f"[TF_GATE_NLI] qid={question_id} nli_label={nli_label_pre} should_complete={should_complete}")

        if not should_complete:
            trace_log["completion_skipped_reason"] = f"tf_nli_{nli_label_pre.lower()}"
            final_context = base_context
        else:
            trace_log["completion_triggered"] = True
            trace_log["completion_reason"] = f"tf_nli_{nli_label_pre.lower()}"

            completion_query = build_tf_completion_query(
                statement=question,
                consensus_gap=trace_log.get("critic_consensus_gap", ""),
            )

            # Hop1 local
            gap_passages, _, _ = retriever.get_retrieved_passages(
                completion_query,
                top_k=5,
                bm25_weight=0.5,
                vector_weight=0.5,
            )

            # Hop1 web
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
                logging.warning(f"Web retrieval failed (Hop1): {e}. Proceeding with local passages only.")
                trace_log["web_query_used"] = ""
                web_passages = []

            merged_passages = deduplicate_passages(base_passages + gap_passages + web_passages)
            merged_context = "\n\n".join(merged_passages[:15])

            # Court bypass
            logging.info("Step 5: Court auditing BYPASSED.")
            trace_log["court_statement_used"] = question.strip()
            trace_log["court_verdict"] = "BYPASSED"
            trace_log["court_flags"] = {}
            trace_log["veto_triggered"] = False
            final_context = merged_context

            # Hop1 post-NLI rerun (must write label)
            try:
                post = rerun_tf_nli(
                    statement=question,
                    context=final_context,
                    retriever=retriever,
                    window_sents=2,
                    max_spans=120,
                    top_k=5,
                )
                nli_post = (post.get("nli") or {})
                trace_log["tf_nli_label_post_completion"] = (nli_post.get("label") or "").strip()
                trace_log["tf_nli_confidence_post_completion"] = nli_post.get("confidence")
                trace_log["tf_nli_best_span_index_post_completion"] = nli_post.get("best_span_index")
                trace_log["tf_nli_citations_post_completion"] = nli_post.get("citations", []) or []
                trace_log["tf_nli_explanation_post_completion"] = nli_post.get("explanation", "") or ""
                trace_log["tf_support_top_spans"] = post.get("top_spans", []) or []
                trace_log["tf_support_top_scores"] = post.get("top_scores", []) or []
            except Exception as e:
                logging.warning(f"Post-completion TF NLI rerun failed (Hop1): {e}")

            hop1_label = _ensure_tf_label(trace_log, "tf_nli_label_post_completion", default="NOT_ENOUGH_INFO")
            logging.info(f"[TF_POST_NLI_HOP1] qid={question_id} label={hop1_label}")

            # Hop2 if still NEI or SOFT
            if hop1_label in ("NOT_ENOUGH_INFO", "SOFT_CONTRADICTION"):
                trace_log["hop2_triggered"] = True
                trace_log["hop2_reason"] = hop1_label
                logging.info(f"[TF_HOP_2] Hop1={hop1_label}. Triggering Hop2 with Query Rewrite...")

                hop2_query = generate_web_rewrite_query(
                    statement=question,
                    gap_hint=trace_log.get("critic_consensus_gap", ""),
                )
                trace_log["web_query_used_hop2"] = hop2_query
                logging.info(f"[TF_HOP_2_QUERY] {hop2_query}")

                hop2_local_passages, _, _ = retriever.get_retrieved_passages(
                    hop2_query,
                    top_k=5,
                    bm25_weight=0.5,
                    vector_weight=0.5,
                )

                web_passages_2: List[str] = []
                try:
                    web_ev_2, _ = research(client=client, target_info=hop2_query, retriever=retriever)
                    web_passages_2 = format_evidence_items(web_ev_2)
                except Exception as e:
                    logging.warning(f"Hop2 Web retrieval failed: {e}")
                    web_passages_2 = []

                if web_passages_2 or hop2_local_passages:
                    merged_passages_2 = deduplicate_passages(merged_passages + hop2_local_passages + web_passages_2)
                    final_context = "\n\n".join(merged_passages_2[:18])

                    try:
                        post2 = rerun_tf_nli(
                            statement=question,
                            context=final_context,
                            retriever=retriever,
                            window_sents=2,
                            max_spans=120,
                            top_k=5,
                        )
                        nli_post2 = (post2.get("nli") or {})
                        trace_log["tf_nli_label_hop2"] = (nli_post2.get("label") or "").strip()
                        trace_log["tf_nli_confidence_hop2"] = nli_post2.get("confidence")
                        trace_log["tf_nli_best_span_index_hop2"] = nli_post2.get("best_span_index")
                        trace_log["tf_nli_citations_hop2"] = nli_post2.get("citations", []) or []
                        trace_log["tf_nli_explanation_hop2"] = nli_post2.get("explanation", "") or ""
                        trace_log["tf_support_top_spans_hop2"] = post2.get("top_spans", []) or []
                        trace_log["tf_support_top_scores_hop2"] = post2.get("top_scores", []) or []
                    except Exception as e:
                        logging.warning(f"Post-completion TF NLI rerun failed (Hop2): {e}")

                hop2_label = _ensure_tf_label(trace_log, "tf_nli_label_hop2", default="NOT_ENOUGH_INFO")
                logging.info(f"[TF_POST_NLI_HOP2] qid={question_id} label={hop2_label}")

        # Injection (audit-only): only if decisive
        final_label_for_inject = _normalize_tf_label(
            (trace_log.get("tf_nli_label_hop2") or "")
            or (trace_log.get("tf_nli_label_post_completion") or "")
            or (trace_log.get("tf_nli_label") or "")
        )
        if final_label_for_inject in ("SUPPORTED", "HARD_CONTRADICTION"):
            use_hop2 = bool(_normalize_tf_label(trace_log.get("tf_nli_label_hop2", "")))
            span = _pick_span_for_injection(trace_log, use_hop2=use_hop2)
            citations = (trace_log.get("tf_nli_citations_hop2") if use_hop2 else trace_log.get("tf_nli_citations_post_completion")) \
                        or trace_log.get("tf_nli_citations") or []
            ev_block = build_tf_top_evidence_block(span=span, nli_label=final_label_for_inject, citations=citations)
            if ev_block:
                final_context = ev_block + "\n" + (final_context or "")
                trace_log["tf_support_injected"] = True

    else:
        # MC/QA: keep existing behavior (lightly cleaned)
        final_context = base_context
        critic_verify_label = (trace_log.get("critic_verify_label") or "").strip().upper()
        consensus_gap_is_negative = bool(trace_log.get("critic_consensus_gap_is_negative"))

        should_complete = (not is_sufficient) and (critic_verify_label == "ABSENT") and (not consensus_gap_is_negative)
        if should_complete:
            trace_log["completion_triggered"] = True
            trace_log["completion_reason"] = "critic_insufficient_and_absent"

            completion_query = f"{question}\n\nMissing info to find: {trace_log.get('missing_info','')}".strip()
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
                web_passages = []

            merged_passages = deduplicate_passages(base_passages + gap_passages + web_passages)
            merged_context = "\n\n".join(merged_passages[:15])

            logging.info("Step 5: Court auditing BYPASSED.")
            trace_log["court_statement_used"] = (trace_log.get("missing_info") or "").strip() or question
            trace_log["court_verdict"] = "BYPASSED"
            trace_log["court_flags"] = {}
            trace_log["veto_triggered"] = False
            final_context = merged_context

   # STEP 6: Final Answer
    logging.info("Step 6: Answer Agent generating final output...")

    if q_type_u == "TF":
        # final label selection priority: hop2 > hop1(post) > pre
        final_nli_label_raw = (
            (trace_log.get("tf_nli_label_hop2") or "")
            or (trace_log.get("tf_nli_label_post_completion") or "")
            or (trace_log.get("tf_nli_label") or "")
        )
        final_nli_label = _normalize_tf_label(final_nli_label_raw)
        
        if final_nli_label not in TF_LABELS:
            final_nli_label = "NOT_ENOUGH_INFO"
            trace_log["tf_final_answer_source"] = "FALLBACK"
        else:
            if _normalize_tf_label(trace_log.get("tf_nli_label_hop2", "")):
                trace_log["tf_final_answer_source"] = "NLI_HOP2"
            elif _normalize_tf_label(trace_log.get("tf_nli_label_post_completion", "")):
                trace_log["tf_final_answer_source"] = "NLI_HOP1"
            else:
                trace_log["tf_final_answer_source"] = "NLI_PRE"

        trace_log["tf_final_nli_label_used"] = final_nli_label

        # -------------------------------------------------------------
        # 混合路由决策：确定性拦截 vs 泛化推理
        # -------------------------------------------------------------
        if final_nli_label == "SUPPORTED":
            locked_answer = "Yes"
            is_locked = True
        elif final_nli_label == "HARD_CONTRADICTION":
            locked_answer = "No"
            is_locked = True
        else:
            # 针对 NOT_ENOUGH_INFO 和 SOFT_CONTRADICTION 解锁
            locked_answer = None
            is_locked = False

        trace_log["tf_answer_llm_locked"] = is_locked

        if is_locked:
            # 走快速/确定性通道：无视上下文，只做强制格式化 (gpt-4o-mini)
            logging.info(f"[TF_FINAL_MAPPING_LOCKED] qid={question_id} label={final_nli_label} -> {locked_answer}")
            final_answer = generate_tf_final_answer_locked(
                client=client,
                question=question,
                context=final_context,
                locked_answer=locked_answer,
                model_name="gpt-4o-mini",
            )
        else:
            # 走推理通道：释放给智能体，带入完整 Context 让其自由推断 (gpt-4o)
            logging.info(f"[TF_FINAL_MAPPING_UNLOCKED] qid={question_id} label={final_nli_label} -> Delegating to LLM reasoning")
            final_answer = generate_final_answer(
                client=client,
                question=question,
                context=final_context,
                q_type=q_type,
                model_name="gpt-4o", 
            )
    else:
        final_answer = generate_final_answer(
            client=client,
            question=question,
            context=final_context,
            q_type=q_type,
            model_name="gpt-4o",
        )

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