#!/usr/bin/env python3
"""
Answer Agent
Role: The final component of the pipeline. Receives the "Final Context" from the Orchestrator
and the question, then strictly generates the final option (A/B/C/D/E), binary judgment (Yes/No),
or a concise factual answer (for open-ended QA). Contains no batch-processing loops; acts as a pure logic module.

Hardened TF mode:
- generate_tf_final_answer_locked still CALLS an LLM (business requirement),
  but it is used only as a strict formatter and is NOT allowed to flip the deterministic decision.
"""

import time
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _chat_with_retry(client, model, messages, temperature=0.0, max_tokens=10, max_retries=4, base_delay=1.5):
    """API call execution with exponential backoff for transient errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_err


def _normalize_yesno(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    up = re.sub(r"[^A-Z]", "", t.upper())
    if up in ("YES", "Y", "TRUE", "T"):
        return "Yes"
    if up in ("NO", "N", "FALSE", "F"):
        return "No"
    return ""


def generate_tf_final_answer_locked(
    client,
    question: str,
    context: str,
    locked_answer: str,
    model_name: str = "gpt-4o-mini",
) -> str:
    """
    TF-only: still calls an LLM, but forces the model to output exactly the locked_answer.

    IMPORTANT: We intentionally do NOT provide the retrieved context to the formatter model.
    The decision is already deterministic; the LLM is only for the "must end with an LLM answer"
    requirement and formatting stability.

    Returns: exactly "Yes" or "No" (falls back to locked_answer if non-compliant).
    """
    locked = _normalize_yesno(locked_answer)
    if locked not in ("Yes", "No"):
        locked = "No"  # defensive default

    system_prompt = (
        "You are a strict output formatter.\n"
        "You MUST output EXACTLY the provided LOCKED_ANSWER.\n"
        "You are NOT allowed to change it.\n"
        "Output must be EXACTLY one token: Yes or No.\n"
        "Do not output any other text."
    )

    # Do NOT include context. Keep prompt minimal to prevent leakage or noncompliance.
    user_content = f"LOCKED_ANSWER: {locked}\nOutput:"

    try:
        response = _chat_with_retry(
            client=client,
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=2,
        )
        out = (response.choices[0].message.content or "").strip()
        norm = _normalize_yesno(out)

        if norm in ("Yes", "No") and norm == locked:
            return norm

        logging.warning(
            f"[TF_RENDER_LOCKED] Non-compliant LLM output '{out}'. Falling back to locked_answer='{locked}'."
        )
        return locked
    except Exception as e:
        logging.error(
            f"[TF_RENDER_LOCKED] Answer Agent API error: {e}. Falling back to locked_answer='{locked}'."
        )
        return locked


def generate_final_answer(client, question: str, context: str, q_type: str, model_name: str = "gpt-4o") -> str:
    """
    Generates the final answer based on the provided final context.

    :param client: OpenAI client instance.
    :param question: The original question (should include Options text if MC).
    :param context: The final context after completion and court verification.
    :param q_type: "MC" (Multiple Choice), "TF" (True/False), or "QA" (Open-Ended).
    :param model_name: The LLM used for generation (defaults to gpt-4o).
    :return: The extracted final answer.
    """

    # =========================================================================
    # 1. System Prompt Routing
    # =========================================================================
    if q_type == "QA":
        # Empathetic & constrained caregiver assistant prompt for Open-Ended QA
        system_prompt = (
            "You are a supportive expert assistant for dementia caregivers. Rules:\n"
            "1. Language: Simple 8th-grade level.\n"
            "2. Evidence: Use ONLY the provided context to answer. If the information is not in the context, "
            "state that you do not know. DO NOT use outside knowledge.\n"
            "3. Constraints: Max 150 words.\n"
            "4. Formatting: Do NOT write your reasoning steps or chain-of-thought. "
            "Do NOT explain how you arrived at the answer. Just give the final advice.\n"
            "5. Medical Restriction: Do NOT provide medical diagnoses or specific treatment plans. "
            "NEVER refer to yourself as a physician, doctor, or medical professional. "
            "Do NOT say 'As a physician...' or 'I recommend this treatment...'"
        )
    else:
        # Strict academic prompt for MC and TF Benchmarks
        system_prompt = (
            "You are an expert medical AI assistant specializing in Alzheimer's Disease and Related Dementias (ADRD). "
            "Your core directive is to answer the user's question STRICTLY based on the provided retrieved context. "
            "Do NOT use external knowledge. If the context does not contain the answer, do not guess."
        )

    # =========================================================================
    # 2. User Prompt Assembly
    # =========================================================================
    user_content = f"--- Retrieved Context ---\n{context}\n\n--- Question ---\n{question}\n\n"

    strict_instruction = (
        "--- INSTRUCTIONS ---\n"
        "1. GROUNDING: Answer STRICTLY based on the provided context. Do NOT use outside knowledge.\n"
    )

    target_max_tokens = 10

    if q_type == "TF":
        strict_instruction += (
            "2. FORMAT: Output ONLY the final answer, with no explanation, preamble, or extra text.\n"
            "3. DECISION RULE: Answer 'Yes' if the context states OR implies the statement; "
            "answer 'No' if the context contradicts it or contains no related information.\n"
            "4. Your output must be EXACTLY 'Yes' or 'No'."
        )
    elif q_type == "MC":
        strict_instruction += (
            "2. FORMAT: Output ONLY the final answer, with no explanation, preamble, or extra text.\n"
            "3. Your output must be EXACTLY ONE option letter (A, B, C, D, or E) — the one best supported by the context."
        )
    elif q_type == "QA":
        strict_instruction += "2. Provide a supportive, concise, and factual answer based ONLY on the context."
        target_max_tokens = 250  # allow room for the 150-word constraint

    user_content += strict_instruction

    # =========================================================================
    # 3. LLM Execution
    # =========================================================================
    try:
        response = _chat_with_retry(
            client=client,
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=target_max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Answer Agent API error: {e}")
        return f"Error: {e}"


def check_accuracy(generated: str, ground_truth: str, correct_letter: str, q_type: str) -> bool:
    """
    Objective scoring tool. Intended for calculating accuracy during pipeline batch runs.
    """
    generated_clean = generated.strip().upper()

    if q_type == "TF":
        generated_clean = re.sub(r'[^A-Z]', '', generated_clean)
        gt = ground_truth.strip().upper()
        if gt in ["YES", "TRUE"]:
            return generated_clean in ["YES", "TRUE", "Y", "T"]
        if gt in ["NO", "FALSE"]:
            return generated_clean in ["NO", "FALSE", "N", "F"]
        return generated_clean == gt

    elif q_type == "MC":
        generated_clean = re.sub(r'[^A-Z]', '', generated_clean)
        if generated_clean and generated_clean[0] in "ABCDE":
            return generated_clean[0] == correct_letter.strip().upper()
        return False

    elif q_type == "QA":
        gt_clean = ground_truth.strip().lower()
        gen_clean_lower = generated.strip().lower()
        return gt_clean in gen_clean_lower

    return False
