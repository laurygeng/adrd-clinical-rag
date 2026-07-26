#!/usr/bin/env python3
"""
Critic Agent (Sufficiency Gate)
Role: Evaluates whether the retrieved context is sufficient to answer the question.
Uses a Dual-Agent (OpenAI + Gemini) Identify-then-Verify (ItV) mechanism.
If sufficient, returns (True, ""). If insufficient, returns (False, "Missing Information").
"""

import os
import time
import logging
import numpy as np
from typing import Tuple, List
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lazily loaded globals for models to avoid overhead on import
_emb_model = None
_oai_client = None
_gem_client = None

def get_emb_model():
    global _emb_model
    if _emb_model is None:
        logging.info("Loading local embedding model for consensus voting...")
        _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb_model

def get_oai_client():
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _oai_client

def get_gem_client():
    global _gem_client
    if _gem_client is None:
        _gem_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _gem_client

# System Prompts tailored by question type
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

SYS_VERIFY = "Decide if a specific piece of information is PRESENT in the context. Answer exactly 'PRESENT' or 'ABSENT'."

def _call_openai(sys_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 40) -> str:
    """Helper to call OpenAI API."""
    client = get_oai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logging.warning(f"OpenAI call failed: {e}")
        return ""

def _call_gemini(sys_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 40) -> str:
    """Helper to call Google Gemini API with basic retry for rate limits."""
    client = get_gem_client()
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{sys_prompt}\n\n{user_prompt}",
                config=types.GenerateContentConfig(
                    temperature=temperature, 
                    max_output_tokens=max_tokens
                )
            )
            return (response.text or "").strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(2 * (attempt + 1))
                continue
            logging.warning(f"Gemini call failed: {e}")
            break
    return ""

def _is_none(text: str) -> bool:
    """Checks if the model determined no missing information."""
    cleaned = text.strip().upper()
    return cleaned.startswith("NONE") or not cleaned

def evaluate_sufficiency(question: str, context: str, q_type: str, calls_per_agent: int = 5) -> Tuple[bool, str]:
    """
    Evaluates context sufficiency using a dual-agent consensus mechanism.
    
    :param question: The question text (including options if MC).
    :param context: The retrieved context passages.
    :param q_type: "MC", "TF", or "QA".
    :param calls_per_agent: Number of times to query each agent for a vote.
    :return: A tuple (is_sufficient: bool, missing_information: str)
    """
    if not context.strip():
        return False, question  # If completely empty, the whole question is missing

    # 1. Select prompt based on type
    if q_type == "TF":
        sys_id = SYS_ID_TF
        user_prompt = f"Context:\n{context}\n\nStatement:\n{question}"
    elif q_type == "MC":
        sys_id = SYS_ID_MC
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    else:
        sys_id = SYS_ID_QA
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    # 2. Identify Phase: Collect votes from both agents
    gaps: List[str] = []
    
    # Run OpenAI and Gemini agents (can be parallelized in production, sequential here for safety)
    for _ in range(calls_per_agent):
        gaps.append(_call_openai(sys_id, user_prompt, temperature=0.7, max_tokens=40))
        gaps.append(_call_gemini(sys_id, user_prompt, temperature=0.7, max_tokens=40))
        
    # 3. Consensus Phase
    none_frac = float(np.mean([_is_none(g) for g in gaps]))
    real_gaps = [g for g in gaps if not _is_none(g)]

    # If the majority (> 60%) say NONE, or if no valid gaps were generated, it is sufficient
    if none_frac >= 0.6 or not real_gaps:
        return True, ""

    # Find the most central "missing info" claim using semantic embeddings
    emb = get_emb_model()
    embeddings = emb.encode(real_gaps, normalize_embeddings=True)
    similarity_matrix = np.dot(embeddings, embeddings.T)
    consensus_idx = int(similarity_matrix.mean(axis=1).argmax())
    consensus_gap = real_gaps[consensus_idx]

    # 4. Verify Phase: Guard against hallucinated gaps
    # Ask one of the agents (OpenAI) to verify if the consensus gap is actually present
    verify_prompt = f"Context:\n{context}\n\nInformation: {consensus_gap}\n\nPresent?"
    verify_result = _call_openai(SYS_VERIFY, verify_prompt, temperature=0.0, max_tokens=10)

    if "PRESENT" in verify_result.upper():
        # The gap was hallucinated by the identify phase; context actually has it
        return True, ""
    else:
        # Gap is truly missing
        return False, consensus_gap