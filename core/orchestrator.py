#!/usr/bin/env python3
"""
Orchestrator (TF Iter-3)

MC/QA: unchanged completion gate:
  (not is_sufficient) AND (critic_verify_label == "ABSENT") AND (not negative_gap)

TF (Iter-3):
- NLI labels: SUPPORTED | HARD_CONTRADICTION | SOFT_CONTRADICTION | NOT_ENOUGH_INFO
- Gate completion by TF NLI:
    NOT_ENOUGH_INFO or SOFT_CONTRADICTION => completion
    SUPPORTED or HARD_CONTRADICTION => no completion
- 2-hop completion:
    Hop1 uses statement + sanitized consensus_gap (local + web)
    If Hop1 post-NLI still NEI or SOFT_CONTRADICTION => Hop2 with query rewrite (local + web)
- TF final answer is deterministic mapping from final TF NLI label (bypass Answer LLM):
    SUPPORTED => Yes
    HARD_CONTRADICTION => No
    SOFT_CONTRADICTION / NOT_ENOUGH_INFO => No (fallback)
- Inject TOP EVIDENCE (TF) ONLY when final NLI is SUPPORTED or HARD_CONTRADICTION
- Court policy: BYPASSED for testing. Veto is never triggered.
"""

import os
import logging
import hashlib
from typing import Dict, Any, List, Union, Optional
from datetime import datetime

from openai import OpenAI

from core.advanced_retriever import AdvancedRetriever
from core.critic_agent import evaluate_sufficiency, rerun_tf_nli
from core.answer_agent import generate_final_answer
from core.search_agent import research
from core.trace_logger import write_jsonl, make_item_id, write_run_meta, get_run_dir

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
    lower = g.lower()
    bad = [
        "single most important piece",
        "still missing",
        "answer exactly",
        "one short phrase",
        "you judge whether",
        "context:",
        "question:",
        "statement:",
        "sufficient to decide",
    ]
    if any(b in lower for b in bad):
        return ""
    if lower in ("none", "n/a", "na", "nan"):
        return ""
    if len(g) > 120 or "\n" in g:
        return ""
    return g


def build_tf_completion_query(statement: str, consensus_gap: str) -> str:
    stmt = (statement or "").strip()
    gap_s = sanitize_gap_text(consensus_gap)
    if gap_s:
        return f"{stmt}\n\nMissing detail to verify: {gap_s}".strip()
    return stmt


def build_tf_top_evidence_block(span: str, nli_label: str, citations: List[str]) -> str:
    span = (span or "").strip()
    if not span:
        return ""
    cite = ""
    if citations:
        cite = "\n\nCITATIONS:\n- " + "\n- ".join([c.strip() for c in citations[:3] if c.strip()])
    return (
        "=== TOP EVIDENCE (TF) ===\n"
        f"[NLI={nli_label}]\n"
        f"{span}\n"
        f"{cite}\n"
        "=== END TOP EVIDENCE (TF) ===\n"
    )


def generate_web_rewrite_query(statement: str, gap_hint: str = "") -> str:
    """
    TF-only query rewrite for Hop2: produce sharp web query keywords.
    Output ONLY the query string.
    """
    client = get_oai_client()
    stmt = (statement or "").strip()
    hint = sanitize_gap_text(gap_hint)
    user = stmt
    if hint:
        user = f"{stmt}\n\nMissing detail to verify: {hint}".strip()

    sys_prompt = (
        "You are an expert search query generator.\n"
        "Convert the given True/False statement into a highly optimized Google search query.\n"
        "Rules:\n"
        "- Extract core medical entities and key relationship.\n"
        "- Preserve any numbers, percentages, age ranges, time ranges.\n"
        "- Include important synonyms.\n"
        "- Drop filler words.\n"
        "- Output ONLY the query string, nothing else."
    )

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=48,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
        )
        q = (r.choices[0].message.content or "").strip()
        return q or stmt
    except Exception:
        return stmt


def _pick_span_for_injection(trace: Dict[str, Any], use_hop2: bool = False) -> str:
    spans = trace.get("tf_support_top_spans_hop2" if use_hop2 else "tf_support_top_spans", []) or []
    best_i = trace.get("tf_nli_best_span_index_hop2" if use_hop2 else "tf_nli_best_span_index_post_completion")
    if best_i is None and not use_hop2:
        best_i = trace.get("tf_nli_best_span_index")  # pre
    if isinstance(best_i, int) and 0 <= best_i < len(spans):
        return spans[best_i]
    return spans[0] if spans else ""


def run_pipeline(question: str, q_type: str, question_id: str = None) -> Dict[str, Any]:
    retriever = get_retriever()
    client = get_oai_client()

    try:
        write_run_meta(
            {
                "run_dir": get_run_dir(),
                "orchestrator_model": "gpt-4o",
                "notes": "Pipeline traces written to logs/<run_ts>/orchestrator/pipeline_traces.jsonl",
            }
        )
    except Exception:
        pass

    item_id = make_item_id(question_id, question)

    trace_log: Dict[str, Any] = {
        "item_id": item_id,
        "question_id": question_id,
        "q_type": q_type,

        "is_sufficient": True,
        "missing_info": "",

        "completion_triggered": False,
        "completion_reason": "",
        "completion_skipped_reason": "",

        "web_query_used": "",
        "web_query_used_hop2": "",
        "hop2_triggered": False,
        "hop2_reason": "",

        "court_statement_used": "",
        "court_verdict": "BYPASSED",
        "court_flags": {},
        "veto_triggered": False,

        # Critic audit
        "critic_decision": "",
        "critic_none_frac_strict": None,
        "critic_empty_frac": None,
        "critic_invalid_gap_frac": None,
        "critic_consensus_gap": "",
        "critic_consensus_gap_is_negative": None,

        "critic_verify_mode": "",
        "critic_verify_label": "",
        "critic_verify_best_score": None,
        "critic_verify_threshold": None,
        "critic_verify_best_span": "",
        "critic_verify_n_spans": None,
        "critic_verify_window_sents": None,
        "critic_md_path": "",

        # TF pre (from critic)
        "tf_support_top_spans": [],
        "tf_support_top_scores": [],
        "tf_nli_label": "",
        "tf_nli_confidence": None,
        "tf_nli_best_span_index": None,
        "tf_nli_citations": [],
        "tf_nli_explanation": "",
        "tf_support_injected": False,

        # TF hop1 post-completion NLI
        "tf_nli_label_post_completion": "",
        "tf_nli_confidence_post_completion": None,
        "tf_nli_best_span_index_post_completion": None,
        "tf_nli_citations_post_completion": [],
        "tf_nli_explanation_post_completion": "",

        # TF hop2 fields (NEW; do not overwrite hop1)
        "tf_nli_label_hop2": "",
        "tf_nli_confidence_hop2": None,
        "tf_nli_best_span_index_hop2": None,
        "tf_nli_citations_hop2": [],
        "tf_nli_explanation_hop2": "",
        "tf_support_top_spans_hop2": [],
        "tf_support_top_scores_hop2": [],

        # TF final decision audit
        "tf_final_nli_label_used": "",
        "tf_final_answer_source": "",  # NLI_PRE / NLI_HOP1 / NLI_HOP2 / FALLBACK
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
        trace_log["tf_nli_explanation"] = critic_trace.get("tf_nli_explanation", "")

    q_type_u = (q_type or "").strip().upper()

    # STEP 3-5
    if q_type_u == "TF":
        nli_label_pre = (trace_log.get("tf_nli_label") or "").strip().upper()

        # Iter-3 gate: NEI or SOFT_CONTRADICTION => completion
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

            # Hop1 post-NLI rerun
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

                logging.info(f"[TF_POST_NLI_HOP1] qid={question_id} label={trace_log['tf_nli_label_post_completion']}")
            except Exception as e:
                logging.warning(f"Post-completion TF NLI rerun failed (Hop1): {e}")

            # ----------------------------
            # ITER-3: Hop2 (Query Rewrite) if still NEI or SOFT_CONTRADICTION
            # ----------------------------
            post_label_1 = (trace_log.get("tf_nli_label_post_completion") or "").strip().upper()
            if post_label_1 in ("NOT_ENOUGH_INFO", "SOFT_CONTRADICTION"):
                trace_log["hop2_triggered"] = True
                trace_log["hop2_reason"] = post_label_1
                logging.info(f"[TF_HOP_2] Hop1={post_label_1}. Triggering Hop2 with Query Rewrite...")

                hop2_query = generate_web_rewrite_query(
                    statement=question,
                    gap_hint=trace_log.get("critic_consensus_gap", ""),
                )
                trace_log["web_query_used_hop2"] = hop2_query
                logging.info(f"[TF_HOP_2_QUERY] {hop2_query}")

                # Hop2 local retrieval (important)
                hop2_local_passages, _, _ = retriever.get_retrieved_passages(
                    hop2_query,
                    top_k=5,
                    bm25_weight=0.5,
                    vector_weight=0.5,
                )

                # Hop2 web retrieval
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

                    # Hop2 NLI rerun
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

                        logging.info(f"[TF_POST_NLI_HOP2] qid={question_id} label={trace_log['tf_nli_label_hop2']}")
                    except Exception as e:
                        logging.warning(f"Post-completion TF NLI rerun failed (Hop2): {e}")

        # ----------------------------
        # TF injection: only if FINAL NLI is decisive (SUPPORTED or HARD_CONTRADICTION)
        # Decide final label source for injection
        # ----------------------------
        nli_label_hop2 = (trace_log.get("tf_nli_label_hop2") or "").strip().upper()
        nli_label_hop1 = (trace_log.get("tf_nli_label_post_completion") or "").strip().upper()
        nli_label_pre_u = nli_label_pre

        final_label_for_inject = nli_label_hop2 or nli_label_hop1 or nli_label_pre_u
        use_hop2_spans = bool(nli_label_hop2)

        if final_label_for_inject in ("SUPPORTED", "HARD_CONTRADICTION"):
            span = _pick_span_for_injection(trace_log, use_hop2=use_hop2_spans)

            citations_final = []
            if use_hop2_spans:
                citations_final = trace_log.get("tf_nli_citations_hop2", []) or []
            else:
                citations_final = trace_log.get("tf_nli_citations_post_completion") or trace_log.get("tf_nli_citations", []) or []

            block = build_tf_top_evidence_block(
                span=span,
                nli_label=final_label_for_inject,
                citations=citations_final,
            )
            if block:
                trace_log["tf_support_injected"] = True
                final_context = block + "\n" + (final_context or "")

    else:
        # MC/QA unchanged gate
        critic_verify_label = (trace_log.get("critic_verify_label") or "").strip()
        consensus_gap_is_negative = bool(trace_log.get("critic_consensus_gap_is_negative"))

        should_complete = (
            (not is_sufficient)
            and (critic_verify_label == "ABSENT")
            and (not consensus_gap_is_negative)
        )

        if not should_complete:
            if is_sufficient:
                trace_log["completion_skipped_reason"] = "sufficient"
            else:
                if consensus_gap_is_negative:
                    trace_log["completion_skipped_reason"] = "negative_gap"
                elif critic_verify_label and critic_verify_label != "ABSENT":
                    trace_log["completion_skipped_reason"] = f"verify_label_{critic_verify_label.lower()}"
                else:
                    trace_log["completion_skipped_reason"] = "verify_missing_or_skipped"
            final_context = base_context
        else:
            trace_log["completion_triggered"] = True
            trace_log["completion_reason"] = "critic_insufficient_and_absent"

            completion_query = f"{question}\n\nMissing info to find: {trace_log['missing_info']}".strip()

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

            # Court bypass
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
        final_nli_label = (
            (trace_log.get("tf_nli_label_hop2") or "")
            or (trace_log.get("tf_nli_label_post_completion") or "")
            or (trace_log.get("tf_nli_label") or "")
        ).strip().upper()

        if trace_log.get("tf_nli_label_hop2"):
            trace_log["tf_final_answer_source"] = "NLI_HOP2"
        elif trace_log.get("tf_nli_label_post_completion"):
            trace_log["tf_final_answer_source"] = "NLI_HOP1"
        else:
            trace_log["tf_final_answer_source"] = "NLI_PRE"

        trace_log["tf_final_nli_label_used"] = final_nli_label

        if final_nli_label == "SUPPORTED":
            final_answer = "Yes"
        elif final_nli_label == "HARD_CONTRADICTION":
            final_answer = "No"
        else:
            # SOFT_CONTRADICTION / NOT_ENOUGH_INFO / unknown
            final_answer = "No"
            if final_nli_label not in ("SOFT_CONTRADICTION", "NOT_ENOUGH_INFO"):
                trace_log["tf_final_answer_source"] = "FALLBACK"

        logging.info(f"[TF_FINAL_MAPPING] qid={question_id} label={final_nli_label} -> {final_answer}")

    else:
        # MC / QA unchanged
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
