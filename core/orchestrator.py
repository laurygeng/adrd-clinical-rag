#!/usr/bin/env python3
"""
Orchestrator (Unified ItV Architecture)

Key hardening:
- All question types (TF, MC, QA) use the unified Identify-then-Verify (ItV) gap mechanism.
- Removed legacy NLI branching.
- Answers are delegated to the LLM (gpt-4o shim) to evaluate the final unified context.
- Added precise step-by-step execution timers for benchmarking.

** ABLATION SUPPORT ADDED **
- Added parameters to selectively disable Base RAG and/or Completion Retrieval.
"""

import os
import time
import logging
import hashlib
from typing import Dict, Any, List, Union
from datetime import datetime

from openai import OpenAI

from core.advanced_retriever import AdvancedRetriever
from core.critic_agent import CRITIC_CALLS_PER_AGENT, evaluate_sufficiency
from core.answer_agent import generate_final_answer
from core.search_agent import clean_search_query_text, research
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
    g = clean_search_query_text(str(gap), fallback="")
    if not g:
        return ""
    return g[:160]


def generate_web_rewrite_query(statement: str, gap_hint: str = "") -> str:
    """精简版复合检索 Query 生成：只提取核心问题主题与缺失事实，严禁包含选项字母和内容"""
    # 过滤掉题干中的 Options 部分，只保留问题本身
    clean_statement = statement.split("Options:")[0].strip()
    gh = sanitize_gap_text(gap_hint)
    
    if not gh:
        return clean_statement[:100]
        
    client = get_oai_client()
    sys_prompt = "You are an expert medical search query generator. Output a concise search query (3-7 keywords)."
    user_prompt = (
        f"Question: {clean_statement}\n"
        f"Missing Fact to Find: {gh}\n\n"
        "Generate a short, precise search query combining the core question subject and the missing fact. "
        "Do NOT include option letters (A, B, C, D, E) or option texts. "
        "Output ONLY the query text without quotes or preamble."
    )
    
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=25,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        query = clean_search_query_text(r.choices[0].message.content or "", fallback=f"{clean_statement} {gh}")
        return query if query else f"{clean_statement} {gh}"
    except Exception as e:
        logging.warning(f"Query rewrite LLM call failed: {e}")
        return f"{clean_statement} {gh}"[:100]


def run_pipeline(question: str, q_type: str = "MC", question_id: str = "", use_rag: bool = True, use_completion: bool = True) -> Dict[str, Any]:
    # Only initialize Retriever if RAG is used to save memory
    retriever = get_retriever() if (use_rag or use_completion) else None
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

        # Execution timers
        "time_base_retrieval": 0.0,
        "time_critic_evaluation": 0.0,
        "time_completion_retrieval": 0.0,
        "time_answer_generation": 0.0,
    }

    # STEP 1: Base Retrieval (Time tracking and ablation control)
    t0 = time.time()
    if use_rag:
        logging.info("Step 1: Running Base Retrieval...")
        base_passages, _, _ = retriever.get_retrieved_passages(
            question,
            top_k=8,
            bm25_weight=0.3,
            vector_weight=0.7,
        )
        base_context = "\n\n".join(base_passages)
    else:
        logging.info("Step 1: SKIPPED (Ablation: No RAG)")
        base_passages = []
        base_context = ""
    trace_log["time_base_retrieval"] = time.time() - t0

    # STEP 2: Critic (Unified) (Time tracking and ablation control)
    t0 = time.time()
    if use_completion:
        logging.info("Step 2: Critic Agent evaluating sufficiency...")
        is_sufficient, missing_info, critic_trace = evaluate_sufficiency(
            question=question,
            context=base_context,
            q_type=q_type,
            calls_per_agent=CRITIC_CALLS_PER_AGENT,
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
    else:
        logging.info("Step 2: SKIPPED (Ablation: No Completion/Critic)")
        is_sufficient = True
        trace_log["is_sufficient"] = True
    trace_log["time_critic_evaluation"] = time.time() - t0

# STEP 3-5: Unified Completion Routing (Time tracking and ablation control)
    final_context = base_context
    critic_verify_label = (trace_log.get("critic_verify_label") or "").strip().upper()
    consensus_gap_is_negative = bool(trace_log.get("critic_consensus_gap_is_negative"))

    # 获取 VerifyLocate 的最高匹配分数
    verify_score = trace_log.get("critic_verify_best_score")
    
    # 只有当分数真正很低（例如 < 0.30，说明本地完全找不到相关支撑句）时，才允许触发外网/多跳补全
    is_truly_absent = (verify_score is not None and verify_score < 0.30)

    # Only trigger completion if completion is enabled and critic deemed it necessary
    should_complete = use_completion and (not is_sufficient) and (critic_verify_label == "ABSENT") and is_truly_absent and (not consensus_gap_is_negative)
    
    t0 = time.time()
    if should_complete:
        trace_log["completion_triggered"] = True
        trace_log["completion_reason"] = "critic_insufficient_and_absent"

        logging.info("Step 3: Triggering Completion Retrieval with Hybrid Question-Gap Query...")
        # [UPGRADED]: 传入原问题和提取的缺口，组合出高精度的复合检索 Query
        completion_query = generate_web_rewrite_query(statement=question, gap_hint=trace_log.get('missing_info', ''))
        trace_log["web_query_used"] = completion_query
        
        # 带着复合 Query 去本地知识库进行二次检索
        gap_passages, _, _ = retriever.get_retrieved_passages(
            completion_query,
            top_k=5,
            bm25_weight=0.6,
            vector_weight=0.4,
        )
        
        base_tagged = [f"[BASE] {p}" for p in base_passages[:3]]
        gap_tagged = [f"[GAP] {p}" for p in gap_passages]

        web_passages_trimmed = []
        try:
            web_evidence, web_query = research(
                client=client,
                target_info=completion_query,
                question=question,
                retriever=retriever,
            )
            web_passages = format_evidence_items(web_evidence)
            web_passages_trimmed = web_passages[:1]  # 仅保留高质量的 1 段外网文本
        except Exception as e:
            logging.warning(f"Web retrieval failed: {e}. Proceeding with local gap passages only.")

      # [UNIFIED REFINEMENT]: 统一架构下的精简补全组合，严格控制上下文噪音上限
        prioritized_passages = base_tagged + web_passages_trimmed + gap_tagged
        merged_passages = deduplicate_passages(prioritized_passages)
        
        # 统一通过 q_type 动态调整最大字符数和最大片段数（保持架构一致的前提下自适应精度）
        is_tf = (q_type.strip().upper() == "TF")
        MAX_CONTEXT_CHARS = 6000 if is_tf else 12000
        max_sources = 3 if is_tf else 6
        
        current_chars = 0
        final_passages = []
        
        for p in merged_passages[:max_sources]:
            if current_chars + len(p) > MAX_CONTEXT_CHARS and len(final_passages) >= 2:
                break
            final_passages.append(p)
            current_chars += len(p)
            
        final_context = "\n\n".join(final_passages)
    else:
        if not use_completion:
            logging.info("Step 3-5: SKIPPED (Ablation: No Completion Retrieval)")
    trace_log["time_completion_retrieval"] = time.time() - t0

    # STEP 6: Final Answer (Unified Delegation to LLM) (Time tracking)
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