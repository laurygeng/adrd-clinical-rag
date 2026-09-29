#!/usr/bin/env python3
"""
Answer Agent

Role: The final component of the pipeline. Receives the "Final Context" from the Orchestrator
and the question, then strictly generates the final option (A/B/C/D/E), binary judgment (Yes/No),
or a concise factual answer (for open-ended QA). Contains no batch-processing loops; acts as a pure logic module.

Hardening:
- Works for instruct/chat models AND base models (routes by tokenizer chat_template capability).
- MC/TF outputs are ALWAYS forced into valid format (A-E / Yes-No) via decision-only retries + parsing.
- No model-name hardcoding (no BioMistral special-case).
"""

import os
import time
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -----------------------------------------------------------------------------
# Low-level call helper
# -----------------------------------------------------------------------------
def _chat_with_retry(client, model, messages, temperature=0.0, max_tokens=20, max_retries=4, base_delay=1.5):
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


# -----------------------------------------------------------------------------
# TF parsing
# -----------------------------------------------------------------------------
def _normalize_yesno(s: str) -> str:
    t = (s or "").strip().upper()
    if not t:
        return ""
    if t in ("YES", "Y", "TRUE", "T"):
        return "Yes"
    if t in ("NO", "N", "FALSE", "F"):
        return "No"
    return ""


def _extract_yesno_answer(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    # 1) Clean direct match
    clean_upper = re.sub(r"[^\w]", "", raw.upper())
    if clean_upper in ("YES", "Y", "TRUE", "T"):
        return "Yes"
    if clean_upper in ("NO", "N", "FALSE", "F"):
        return "No"

    # 2) Common prefix formats
    match = re.search(r"(?i)(?:answer|output|result|decision)[\s:]*(yes|no|true|false)", raw)
    if match:
        val = match.group(1).upper()
        return "Yes" if val in ("YES", "TRUE") else "No"

    # 3) First isolated token
    match = re.search(r"(?i)\b(yes|no|true|false)\b", raw)
    if match:
        val = match.group(1).upper()
        return "Yes" if val in ("YES", "TRUE") else "No"

    # 4) Small-model phrasing heuristics
    lowered = raw.lower()
    negative_markers = [
        "not supported by the context",
        "not supported by the provided context",
        "contradicted by the context",
        "the statement is false",
        "this statement is false",
        "inconsistent with the context",
        "the context does not support",
    ]
    positive_markers = [
        "supported by the context",
        "supported by the provided context",
        "is supported by the context",
        "the statement is supported",
        "the statement is true",
        "this statement is true",
        "consistent with the context",
    ]
    if any(m in lowered for m in negative_markers):
        return "No"
    if any(m in lowered for m in positive_markers):
        return "Yes"

    return ""


# -----------------------------------------------------------------------------
# MC parsing
# -----------------------------------------------------------------------------
def _extract_mc_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in str(question or "").splitlines():
        match = re.match(r"\s*([A-E])\.\s*(.+?)\s*$", line)
        if match:
            options[match.group(1).upper()] = match.group(2).strip()
    return options


def _extract_mc_answer(text: str, question: str = "") -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    # 0) Numeric-only output (common for base models): 1-5 => A-E
    m = re.fullmatch(r"\s*([1-5])\s*", raw)
    if m:
        return "ABCDE"[int(m.group(1)) - 1]

    # 1) Extremely clean direct match: "A", "(B)", "Option C"
    match = re.fullmatch(r"(?i)\(?option\s+([A-E])\)?|\(?([A-E])\)?\.?", raw)
    if match:
        return (match.group(1) or match.group(2)).upper()

    # 2) Typical verbose prefix: "Answer: C"
    match = re.search(r"(?i)(?:answer|option|output|choice)(?:\s+is)?[\s=:\*]*\(?([A-E])\)?\b", raw)
    if match:
        return match.group(1).upper()

    # 3) Starts with a letter: "**B** because..."
    match = re.match(r"^\s*\**([A-E])\**\b", raw)
    if match:
        return match.group(1).upper()

    # 4) Search formatted occurrences (avoid article "A" false positives by requiring formatting)
    match = re.search(r"(?i)\boption\s+([A-E])\b|\b([A-E])\.(?=\s|$)|(?<=\s)\(([A-E])\)", raw)
    if match:
        return (match.group(1) or match.group(2) or match.group(3)).upper()

    # 5) If model output contains option text instead of letter, map back
    normalized_raw = re.sub(r"\s+", " ", raw).strip().lower()
    options = _extract_mc_options(question)
    for letter, option_text in options.items():
        option_norm = re.sub(r"\s+", " ", option_text).strip().lower()
        if not option_norm:
            continue
        if option_norm in normalized_raw or normalized_raw in option_norm:
            return letter

    # 6) "none of the above" style
    none_match = re.search(r"(?i)none of the options.*correct", raw)
    if none_match:
        for letter, option_text in options.items():
            option_norm = option_text.lower()
            if any(k in option_norm for k in ["none of the above", "all of the above"]):
                return letter

    return ""


# -----------------------------------------------------------------------------
# Capability routing
# -----------------------------------------------------------------------------
def _detect_has_chat_template(model_name: str) -> bool:
    """Best-effort capability detection.

    Reuse llm_engine's tokenizer; if anything fails, default to base-mode (more robust for small/base models).
    """
    try:
        from core.llm_engine import _get_model_and_tokenizer, _has_chat_template
        _, tokenizer = _get_model_and_tokenizer(model_name)
        return bool(_has_chat_template(tokenizer))
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Decision-only retries (MC/TF)
# -----------------------------------------------------------------------------
def _mc_decision_only_call(client, target_model: str, context: str, question: str, max_tokens: int = 4) -> str:
    """Second-stage MC: force the model to output ONLY one letter."""
    prompt = (
        "You are a multiple-choice answer selector.\n"
        "Return ONLY a single letter: A, B, C, D, or E.\n"
        "Do NOT output words, numbers, punctuation, or whitespace.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    r = _chat_with_retry(
        client=client,
        model=target_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=3,
    )
    return (r.choices[0].message.content or "").strip()


def _tf_decision_only_call(client, target_model: str, context: str, question: str, max_tokens: int = 4) -> str:
    """Second-stage TF: force ONLY Yes/No."""
    prompt = (
        "Return ONLY one token: Yes or No.\n"
        "Do NOT output any other text.\n\n"
        f"Context:\n{context}\n\n"
        f"Statement:\n{question}\n\n"
        "Answer:"
    )
    r = _chat_with_retry(
        client=client,
        model=target_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=3,
    )
    return (r.choices[0].message.content or "").strip()


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def generate_final_answer(client, question: str, context: str, q_type: str, model_name: str = None) -> str:
    """Generates the final answer based on the provided final context."""
    target_model = model_name or os.environ.get("GLOBAL_AGENT_MODEL", "gpt-4o")
    q_type_u = str(q_type or "").strip().upper()

    has_chat = _detect_has_chat_template(target_model)

    # Keep system prompt minimal; base models may ignore it anyway.
    if q_type_u == "QA":
        system_prompt = (
            "You are a supportive expert assistant for dementia caregivers. "
            "Answer ONLY using the provided context. If missing, say you don't know. "
            "Max 150 words."
        )
    else:
        system_prompt = "Answer STRICTLY using the provided context. Do not use outside knowledge."

    context_block = f"--- Retrieved Context ---\n{context}\n\n"

    # -------------------------
    # Prompt routing
    # -------------------------
    if not has_chat:
        # Base-model channel: completion-friendly minimal formatting.
        if q_type_u == "TF":
            user_content = (
                f"{context_block}"
                f"Statement: {question}\n"
                "Answer Yes or No.\n"
                "Answer:"
            )
            target_max_tokens = 16
        elif q_type_u == "MC":
            user_content = (
                f"{context_block}"
                f"Question: {question}\n"
                "Answer with exactly ONE letter: A, B, C, D, or E.\n"
                "Answer:"
            )
            target_max_tokens = 16
        else:
            user_content = (
                f"{context_block}"
                f"Question: {question}\n"
                "Answer concisely using only the context.\n"
                "Answer:"
            )
            target_max_tokens = 240
    else:
        # Instruct/chat channel: richer constraints + small examples.
        instructions = (
            "--- INSTRUCTIONS ---\n"
            "1. GROUNDING: Answer STRICTLY based on the provided context above. Do NOT use outside knowledge.\n"
        )
        if q_type_u == "TF":
            instructions += (
                "2. FORMAT: Output ONLY 'Yes' or 'No'. No explanation.\n"
                "--- EXAMPLES ---\n"
                "Question: True or False statement: All apples are red.\n"
                "Final Answer: No\n\n"
                "Question: True or False statement: Physical activity is beneficial for cardiovascular health.\n"
                "Final Answer: Yes\n\n"
            )
            target_max_tokens = 20
        elif q_type_u == "MC":
            instructions += (
                "2. FORMAT: Output EXACTLY ONE option letter (A, B, C, D, or E). No explanation.\n\n"
                "--- EXAMPLES ---\n"
                "Question: What color can apples be?\n"
                "Options:\n"
                "  A. Blue\n"
                "  B. Red and Green\n"
                "  C. Yellow\n"
                "Final Answer: B\n\n"
            )
            target_max_tokens = 30
        else:
            instructions += "2. Provide a concise grounded answer.\n"
            target_max_tokens = 260

        user_content = (
            f"{context_block}"
            f"{instructions}"
            f"--- YOUR TASK ---\n"
            f"Question: {question}\n"
            "Final Answer: "
        )

    # -------------------------
    # LLM execution + strict parsing + decision-only retries
    # -------------------------
    def _call(max_tokens: int):
        return _chat_with_retry(
            client=client,
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

    try:
        response = _call(target_max_tokens)
        raw_output = (response.choices[0].message.content or "").strip()

        # ---- TF ----
        if q_type_u == "TF":
            parsed = _extract_yesno_answer(raw_output)
            if parsed:
                return parsed

            # Decision-only retry 1
            raw2 = _tf_decision_only_call(client, target_model, context=context, question=question, max_tokens=4)
            parsed2 = _extract_yesno_answer(raw2)
            if parsed2:
                return parsed2

            # Decision-only retry 2 (even shorter)
            raw3 = _tf_decision_only_call(client, target_model, context=context, question=question, max_tokens=2)
            parsed3 = _extract_yesno_answer(raw3)
            if parsed3:
                return parsed3

            logging.warning(f"[TF_PARSE_FAIL] raw='{raw_output}' raw2='{raw2}' raw3='{raw3}'")
            return "No"

        # ---- MC ----
        if q_type_u == "MC":
            parsed = _extract_mc_answer(raw_output, question=question)
            if parsed:
                return parsed

            # Decision-only retry 1
            raw2 = _mc_decision_only_call(client, target_model, context=context, question=question, max_tokens=4)
            parsed2 = _extract_mc_answer(raw2, question=question)
            if parsed2:
                return parsed2

            # Decision-only retry 2 (even shorter)
            raw3 = _mc_decision_only_call(client, target_model, context=context, question=question, max_tokens=2)
            parsed3 = _extract_mc_answer(raw3, question=question)
            if parsed3:
                return parsed3

            logging.warning(f"[MC_PARSE_FAIL] raw='{raw_output}' raw2='{raw2}' raw3='{raw3}'")
            return "A"

        # ---- QA ----
        if raw_output:
            return raw_output

        # QA retry with more budget
        response2 = _call(max(96, target_max_tokens))
        raw2 = (response2.choices[0].message.content or "").strip()
        return raw2 or "I don't know based on the provided context."

    except Exception as e:
        logging.error(f"Answer Agent API error: {e}")
        if q_type_u == "TF":
            return "No"
        if q_type_u == "MC":
            return "A"
        return f"Error: {e}"


def check_accuracy(generated: str, ground_truth: str, correct_letter: str, q_type: str) -> bool:
    """Objective scoring tool. Intended for calculating accuracy during pipeline batch runs."""
    generated_clean = (generated or "").strip().upper()
    if not generated_clean:
        return False

    q_type_u = str(q_type or "").strip().upper()

    if q_type_u == "TF":
        gt = (ground_truth or "").strip().upper()
        if gt in ["YES", "TRUE"]:
            return generated_clean in ["YES", "Y", "TRUE", "T"]
        if gt in ["NO", "FALSE"]:
            return generated_clean in ["NO", "N", "FALSE", "F"]
        return generated_clean == gt

    if q_type_u == "MC":
        # Extractor guarantees single letter if working; fallback returns A anyway.
        return generated_clean == (correct_letter or "").strip().upper()

    if q_type_u == "QA":
        gt_clean = (ground_truth or "").strip().lower()
        return gt_clean in generated_clean.lower()

    return False
