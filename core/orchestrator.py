#!/usr/bin/env python3
"""
Orchestrator
Role: The central nervous system of the RAG pipeline. It coordinates:
1. Base Retrieval
2. Critic Agent (Sufficiency Check)
3. Gap-Guided Local & Web Completion (if needed)
4. Verification Agent (Veto mechanism)
5. Final Answer Generation

Fully compatible with Batch Processing (TF, MC, and QA types).
"""

import os
import logging
import hashlib
from typing import Dict, Any

from openai import OpenAI

# ---------------------------------------------------------
# Component Imports (ensure these files exist in core/)
# ---------------------------------------------------------
from core.advanced_retriever import AdvancedRetriever
from core.critic_agent import evaluate_sufficiency
from core.answer_agent import generate_final_answer
from core.search_agent import research
from core.verification_agent import court

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize global clients/resources to avoid reloading them on every question
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

def deduplicate_passages(passages: list) -> list:
    """Removes exact or near-duplicate passages to maximize context window utility."""
    seen_hashes = set()
    unique_passages = []
    for p in passages:
        if not p:
            continue
        # Use first 100 characters to catch slight variations of the same paragraph
        chunk_hash = hashlib.md5(p.strip()[:100].encode('utf-8')).hexdigest()
        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            unique_passages.append(p)
    return unique_passages

def run_pipeline(question: str, q_type: str) -> Dict[str, Any]:
    """
    Executes the full Retrieval-Critic-Completion-Verification-Answer pipeline.
    
    :param question: The full question string (including options if MC).
    :param q_type: "TF", "MC", or "QA".
    :return: A dictionary containing the final answer and tracing metadata.
    """
    retriever = get_retriever()
    client = get_oai_client()
    
    trace_log = {
        "is_sufficient": True,
        "missing_info": "",
        "completion_triggered": False,
        "court_verdict": "N/A",
        "veto_triggered": False
    }

    # =========================================================================
    # STEP 1: Base Retrieval Agent
    # =========================================================================
    logging.info("Step 1: Running Base Retrieval...")
    base_passages, _, _ = retriever.get_retrieved_passages(
        question, 
        top_k=8, 
        bm25_weight=0.3, 
        vector_weight=0.7
    )
    base_context = "\n\n".join(base_passages)

    # =========================================================================
    # STEP 2: Critic Agent (Sufficiency Evaluation)
    # =========================================================================
    logging.info("Step 2: Critic Agent evaluating sufficiency...")
    is_sufficient, missing_info = evaluate_sufficiency(
        question=question, 
        context=base_context, 
        q_type=q_type, 
        calls_per_agent=5
    )
    
    trace_log["is_sufficient"] = is_sufficient
    trace_log["missing_info"] = missing_info

    if is_sufficient:
        logging.info(" -> Context is SUFFICIENT. Proceeding directly to Answer Agent.")
        final_context = base_context
    else:
        # =====================================================================
        # STEP 3: Auto-Completion (Local + Web)
        # =====================================================================
        logging.info(f" -> Context is INSUFFICIENT. Missing: '{missing_info}'. Triggering Auto-Completion.")
        trace_log["completion_triggered"] = True
        
        # 3A. Gap-guided Local Retrieval
        gap_passages, _, _ = retriever.get_retrieved_passages(
            missing_info, 
            top_k=5, 
            bm25_weight=0.5, 
            vector_weight=0.5
        )
        
        # 3B. Web Retrieval (Single targeted query based on missing info)
        try:
            web_passages, web_query = research(client=client, target_info=missing_info, retriever=retriever)
        except Exception as e:
            logging.warning(f"Web retrieval failed: {e}. Proceeding with local gap passages only.")
            web_passages = []

        # =====================================================================
        # STEP 4: Merge & Deduplicate
        # =====================================================================
        # Keep original base passages at the top to anchor the knowledge, then append new evidence
        merged_passages = deduplicate_passages(base_passages + gap_passages + web_passages)
        merged_context = "\n\n".join(merged_passages[:15])  # Cap at 15 to prevent context window overflow

        # =====================================================================
        # STEP 5: Verification Agent (Court)
        # =====================================================================
        logging.info("Step 5: Verification Agent auditing completed evidence...")
        try:
            # The Court checks if the new merged evidence introduces hallucinated/misaligned concepts
            verdict, flags = court(statement=question, evidence=merged_context, use_mesh=True)
            trace_log["court_verdict"] = verdict
            
            if verdict == "INSUFFICIENT":
                logging.warning(" -> VETO TRIGGERED: Court rejected the completed evidence (Entity/Modal mismatch). Reverting to base context.")
                trace_log["veto_triggered"] = True
                final_context = base_context  # Discard toxic/misaligned completions
            else:
                logging.info(" -> Court APPROVED the completed evidence.")
                final_context = merged_context # Accept the enriched context
                
        except Exception as e:
            logging.error(f"Court evaluation encountered an error: {e}. Defaulting to safe base context.")
            final_context = base_context

    # =========================================================================
    # STEP 6: Answer Agent (Generation)
    # =========================================================================
    logging.info("Step 6: Answer Agent generating final output...")
    final_answer = generate_final_answer(
        client=client,
        question=question,
        context=final_context,
        q_type=q_type,
        model_name="gpt-4o"
    )

    return {
        "final_answer": final_answer,
        "final_context": final_context,
        "trace": trace_log
    }

# Quick test execution block (ignored when imported by the batch runner)
if __name__ == "__main__":
    sample_question = "What are the common early symptoms of Alzheimer's Disease? Options: A. Memory loss B. Heart palpitations C. Hair loss"
    sample_type = "MC"
    
    print("Testing Orchestrator Pipeline...")
    result = run_pipeline(question=sample_question, q_type=sample_type)
    
    print("\n--- FINAL ANSWER ---")
    print(result["final_answer"])
    print("\n--- TRACE LOG ---")
    print(result["trace"])