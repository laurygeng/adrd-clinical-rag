#!/usr/bin/env python3
"""
Orchestrator (Unified ItV Architecture & Multi-Agent Pipeline)

Key hardening:
- All question types (TF, MC, QA) use the unified Identify-then-Verify (ItV) gap mechanism.
- Removed legacy NLI branching.
- Answers and validations are delegated to the Unified Local LLM Engine.
- Context assembly is fully offloaded to Integration Agent.
- Added precise step-by-step execution timers for benchmarking.

** ABLATION SUPPORT ADDED **
- Parameters to selectively disable Base RAG and/or Completion Retrieval.
- Global model routing via GLOBAL_AGENT_MODEL environment variable.
"""

import os
import time
import logging
from typing import Dict, Any
from datetime import datetime

from core.llm_engine import LocalLLMClient
from core.advanced_retriever import AdvancedRetriever
from core.critic_agent import CRITIC_CALLS_PER_AGENT, evaluate_sufficiency
from core.search_agent import clean_search_query_text, research
from core.answer_agent import generate_final_answer
from core.integration_agent import assemble_context
from core.trace_logger import write_jsonl, make_item_id, get_run_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_retriever_instance = None
_llm_client = None


def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        logging.info("Initializing AdvancedRetriever...")
        _retriever_instance = AdvancedRetriever()
    return _retriever_instance


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLLMClient()
    return _llm_client


def sanitize_gap_text(gap: str) -> str:
    if gap is None:
        return ""
    g = clean_search_query_text(str(gap), fallback="")
    if not g:
        return ""
    return g[:160]


def generate_web_rewrite_query(statement: str, gap_hint: str = "", q_type: str = "MC", model_name: str = "gpt-4o") -> str:
    """精简版复合检索 Query 生成：根据题型(QA vs MC/TF)隔离 Prompt，防止泛化导致召回偏差"""
    clean_statement = statement.split("Options:")[0].strip()
    gh = sanitize_gap_text(gap_hint)
    
    if not gh:
        return clean_statement[:100]
        
    client = get_llm_client()
    sys_prompt = "You are an expert medical search query generator. Output a concise search query (3-7 keywords)."
    
    if str(q_type).strip().upper() == "QA":
        # QA 专属 Prompt：严禁泛化，必须保留具体实体
        user_prompt = (
            f"Question: {clean_statement}\n"
            f"Missing Fact to Find: {gh}\n\n"
            "Generate a highly specific search query (3-7 keywords). "
            "CRITICAL: You MUST retain the exact medical conditions, interventions (e.g. secondhand smoke), "
            "or specific scenarios mentioned in the question. Do NOT generalize the topic into broad guidelines. "
            "Output ONLY the query text without quotes or preamble."
        )
    else:
        # MC/TF 保持原有的泛化 Prompt
        user_prompt = (
            f"Question: {clean_statement}\n"
            f"Missing Fact to Find: {gh}\n\n"
            "Generate a short, precise search query (3-7 keywords). "
            "CRITICAL RULE: Extract the broad medical concept, intervention, or clinical topic. "
            "Do NOT include the specific assertions, restrictive behaviors, or exact claims made in the question, "
            "as those may be the 'myths' or 'false statements' being tested. Formulate the query to find the general factual baseline or standard best practices for the topic.\n"
            "Do NOT include option letters (A, B, C, D, E). "
            "Output ONLY the query text without quotes or preamble."
        )
    
    try:
        r = client.chat.completions.create(
            model=model_name,
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
    retriever = get_retriever() if (use_rag or use_completion) else None
    client = get_llm_client()
    
    # 动态获取全局统一的测试模型名称
    global_model = os.environ.get("GLOBAL_AGENT_MODEL", "gpt-4o")

    run_dir = get_run_dir()
    item_id = make_item_id(question_id, question)
    is_qa = (q_type.strip().upper() == "QA")
    is_tf = (q_type.strip().upper() == "TF")

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

    # STEP 1: Base Retrieval
    t0 = time.time()
    
    # clean_query = question.split("Options:")[0].strip() if "Options:" in question else question
    
    # if use_rag:
    #     logging.info("Step 1: Running Base Retrieval...")
    #     base_passages, _, _ = retriever.get_retrieved_passages(
    #         clean_query,  # 【关键修改】：这里将 question 替换为 clean_query
    #         top_k=8,
    #         bm25_weight=0.3,
    #         vector_weight=0.7,
    #     )
        
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

    # STEP 2: Critic Evaluation
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

    # STEP 3 & 4: Unified Completion Routing & Integration
    final_context = base_context
    critic_verify_label = (trace_log.get("critic_verify_label") or "").strip().upper()
    consensus_gap_is_negative = bool(trace_log.get("critic_consensus_gap_is_negative"))

    if is_qa:
        # QA 专属激进补全条件：只要不充分就立即补全
        should_complete = use_completion and (not is_sufficient)
    else:
        should_complete = use_completion and (not is_sufficient) and (critic_verify_label == "ABSENT") and (not consensus_gap_is_negative)
    
    t0 = time.time()
    if should_complete:
        trace_log["completion_triggered"] = True
        trace_log["completion_reason"] = "critic_insufficient_and_absent"

        logging.info("Step 3: Triggering Completion Retrieval with Hybrid Question-Gap Query...")
        completion_query = generate_web_rewrite_query(statement=question, gap_hint=trace_log.get('missing_info', ''), q_type=q_type, model_name=global_model)
        trace_log["web_query_used"] = completion_query
        
        # 3a. Local Gap Retrieval
        gap_passages, _, _ = retriever.get_retrieved_passages(
            completion_query,
            top_k=5,
            bm25_weight=0.6,
            vector_weight=0.4,
        )

        # 3b. Web Search & Refinement
        web_evidence = []
        try:
            web_evidence, web_query, search_log_md = research(
                client=client,
                target_info=completion_query,
                question=question,
                retriever=retriever,
                q_type=q_type,
                model_name=global_model
            )
            
            md_path = trace_log.get("critic_md_path")
            if md_path and os.path.exists(md_path):
                with open(md_path, "a", encoding="utf-8") as f:
                    f.write(search_log_md)
        except Exception as e:
            logging.warning(f"Web retrieval failed: {e}. Proceeding with local gap passages only.")

        # 4. Context Assembly (Integration Agent)
        logging.info("Step 4: Integration Agent assembling cohesive context...")
        final_context = assemble_context(
            base_passages=base_passages,
            gap_passages=gap_passages,
            web_evidence=web_evidence,
            q_type=q_type
        )
    else:
        if not use_completion:
            logging.info("Step 3-4: SKIPPED (Ablation: No Completion Retrieval)")
    trace_log["time_completion_retrieval"] = time.time() - t0

    # STEP 5: Final Answer
    logging.info("Step 5: Answer Agent generating final output...")
    t0 = time.time()
    final_answer = generate_final_answer(
        client=client,
        question=question,
        context=final_context,
        q_type=q_type,
        model_name=global_model,
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