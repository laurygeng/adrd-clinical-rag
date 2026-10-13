#!/usr/bin/env python3
"""
Orchestrator (Unified ItV Architecture & Multi-Agent Pipeline)
已集成：异步 Web 证据自动归档 (Automated Web Knowledge Archiving)
"""

import os
import re
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

from core.llm_engine import LocalLLMClient, _get_model_and_tokenizer, _has_chat_template
from core.advanced_retriever import AdvancedRetriever
from core.critic_agent import CRITIC_CALLS_PER_AGENT, evaluate_sufficiency
from core.search_agent import clean_search_query_text, research
from core.answer_agent import generate_final_answer
from core.integration_agent import assemble_context
from core.trace_logger import write_jsonl, make_item_id, get_run_dir

# 导入自动归档脚本
try:
    from core.archive_web_context_to_kb import archive_evidence_records
    ARCHIVE_AVAILABLE = False # 测试版本
except ImportError:
    ARCHIVE_AVAILABLE = False
    logging.warning("⚠️ archive_web_context_to_kb module not found. Web archiving disabled.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _dbg_enabled() -> bool:
    return os.environ.get('HULU_DEBUG', '1') == '1'

def _clip_text(x: str, limit: int = 1000) -> str:
    s = str(x or '')
    return s if len(s) <= limit else s[:limit] + f' ...[truncated {len(s) - limit} chars]'

def _dbg(stage: str, **kwargs) -> None:
    if not _dbg_enabled(): return
    payload = ' | '.join(f'{k}={_clip_text(v)}' for k, v in kwargs.items())
    logger.warning(f'[HULU-DEBUG][orchestrator][{stage}] {payload}')

_retriever_instance = None
_llm_client = None

def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        logging.info('Initializing AdvancedRetriever...')
        _retriever_instance = AdvancedRetriever()
    return _retriever_instance

def get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLLMClient()
    return _llm_client

def _safe_tag(s: str) -> str:
    s = (s or '').strip()
    if not s: return 'unknown'
    tag = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '-' for ch in s).strip('-_')
    return tag[:80]

def sanitize_gap_text(gap: str) -> str:
    g = clean_search_query_text(str(gap or ''), fallback='')
    return g[:160] if g else ''

def generate_web_rewrite_query(statement: str, gap_hint: str = '', q_type: str = 'MC', model_name: str = 'gpt-4o') -> str:
    clean_statement = statement.split('Options:')[0].strip()
    gh = sanitize_gap_text(gap_hint)
    if not gh: return clean_statement[:100]

    client = get_llm_client()
    sys_prompt = 'You are an expert medical search query generator. Output a concise search query (3-7 keywords).'
    
    if str(q_type).strip().upper() == 'QA':
        user_prompt = f'Question: {clean_statement}\nMissing Fact to Find: {gh}\n\nGenerate a highly specific search query (3-7 keywords). CRITICAL: You MUST retain the exact medical conditions, interventions, or specific scenarios mentioned in the question. Do NOT generalize the topic into broad guidelines. Output ONLY the query text without quotes or preamble.'
    else:
        user_prompt = (
                    f'Question: {clean_statement}\n'
                    f'Missing Fact to Find: {gh}\n\n'
                    'Generate a short, precise search query (3-6 keywords). '
                    'CRITICAL RULE: Extract the core clinical scenario, specific communication phrase, or specific caregiving dilemma from the question. '
                    'Generate a query targeting this exact specific scenario. '
                    'Do NOT over-generalize the topic into broad guidelines. '
                    'Strictly avoid the specific option letters (A, B, C, D, E). '
                    'Output ONLY the query text without quotes or preamble.'
                )
    try:
        r = client.chat.completions.create(model=model_name, temperature=0.0, max_tokens=25, messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_prompt}])
        query = clean_search_query_text(r.choices[0].message.content or '', fallback=f'{clean_statement} {gh}')
        return query if query else f'{clean_statement} {gh}'
    except Exception:
        return f'{clean_statement} {gh}'[:100]

# --- 异步自动归档函数 ---
def _trigger_async_archiving(question_id: str, question: str, retrieval_query: str, q_type: str, web_evidence: list):
    if not ARCHIVE_AVAILABLE or not web_evidence:
        return

    def archive_task():
        try:
            base_dir = Path(__file__).resolve().parents[1] / "knowledge_base" / "raw_files" / "web_external"
            record = {
                "question_id": question_id,
                "question": question,
                "retrieval_query": retrieval_query,
                "q_type": q_type,
                "evidence": web_evidence
            }
            # 这里是纯文本写硬盘操作，速度极快，且因为在独立线程，绝不阻塞你的系统
            stats = archive_evidence_records([record], base_dir)
            if stats.get('items_saved', 0) > 0:
                logging.info(f"✅ Async Archiving: Saved {stats['items_saved']} new external docs to {base_dir}")
        except Exception as e:
            logging.error(f"❌ Async Archiving Failed: {e}")

    # 启动守护线程
    threading.Thread(target=archive_task, daemon=True).start()

def run_pipeline(question: str, q_type: str = 'MC', question_id: str = '', use_rag: bool = True, use_completion: bool = True) -> Dict[str, Any]:
    retriever = get_retriever() if (use_rag or use_completion) else None
    client = get_llm_client()

    global_model = os.environ.get('GLOBAL_AGENT_MODEL', 'gpt-4o')
    model_tag = _safe_tag(global_model)
    qtype_tag = _safe_tag(q_type)

    run_dir = get_run_dir()
    item_id = make_item_id(question_id, question)
    is_qa = (q_type.strip().upper() == 'QA')

    _dbg('pipeline_begin', question_id=question_id, q_type=q_type, model_name=global_model)

    trace_log: Dict[str, Any] = {'q_type': q_type, 'model_name': global_model}

    # STEP 1: Base Retrieval
    t0 = time.time()
    if use_rag:
        logging.info('Step 1: Running Base Retrieval...')
        base_passages, _, _ = retriever.get_retrieved_passages(question, top_k=8, bm25_weight=0.3, vector_weight=0.7)
        base_context = '\n\n'.join(base_passages)
    else:
        base_passages, base_context = [], ''
    trace_log['time_base_retrieval'] = time.time() - t0

    # STEP 2: Critic Evaluation
    t0 = time.time()
    if use_completion:
        logging.info('Step 2: Critic Agent evaluating sufficiency...')
        is_sufficient, missing_info, critic_trace = evaluate_sufficiency(
            question=question, context=base_context, q_type=q_type, calls_per_agent=CRITIC_CALLS_PER_AGENT, question_id=question_id, retriever=retriever)
        trace_log['is_sufficient'] = is_sufficient
        trace_log['missing_info'] = (missing_info or '').strip()
        if isinstance(critic_trace, dict):
            trace_log.update({k: critic_trace.get(k) for k in ['verify_label', 'consensus_gap_is_negative', 'critic_md_path']})
    else:
        is_sufficient = True
        trace_log['is_sufficient'] = True
    trace_log['time_critic_evaluation'] = time.time() - t0

    # STEP 3 & 4: Completion Routing & Integration
    final_context = base_context
    critic_verify_label = (trace_log.get('verify_label') or '').strip().upper()
    consensus_gap_is_negative = bool(trace_log.get('consensus_gap_is_negative'))
    should_complete = use_completion and (not is_sufficient) if is_qa else use_completion and (not is_sufficient) and (critic_verify_label == 'ABSENT') and (not consensus_gap_is_negative)

    t0 = time.time()
    if should_complete:
        logging.info('Step 3: Triggering Completion Retrieval with Hybrid Question-Gap Query...')
        completion_query = generate_web_rewrite_query(statement=question, gap_hint=trace_log.get('missing_info', ''), q_type=q_type, model_name=global_model)
        gap_passages, _, _ = retriever.get_retrieved_passages(completion_query, top_k=5, bm25_weight=0.6, vector_weight=0.4)

        web_evidence = []
        try:
            web_evidence, web_query, search_log_md = research(client=client, target_info=completion_query, question=question, retriever=retriever, q_type=q_type, model_name=global_model)
            
            md_path = trace_log.get('critic_md_path')
            if md_path and os.path.exists(md_path):
                with open(md_path, 'a', encoding='utf-8') as f: f.write(search_log_md)
                
            # 💡 新增：触发后台自动归档
            if web_evidence:
                _trigger_async_archiving(
                    question_id=item_id, 
                    question=question, 
                    retrieval_query=completion_query, 
                    q_type=q_type, 
                    web_evidence=web_evidence
                )
                
        except Exception as e:
            logging.warning(f'Web retrieval failed: {e}. Proceeding with local gap passages only.')

        logging.info('Step 4: Integration Agent assembling cohesive context...')
        final_context = assemble_context(base_passages=base_passages, gap_passages=gap_passages, web_evidence=web_evidence, q_type=q_type)
    trace_log['time_completion_retrieval'] = time.time() - t0

    # STEP 5: Final Answer
    logging.info('Step 5: Answer Agent generating final output...')
    final_answer = generate_final_answer(client=client, question=question, context=final_context, q_type=q_type, model_name=global_model)

    return {'final_answer': final_answer, 'final_context': final_context, 'trace': trace_log}