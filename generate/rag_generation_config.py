#!/usr/bin/env python3
"""
Shared generation configuration for ADRD answer scripts.
This file is intentionally separated from retrieval configuration.
"""

from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are an expert on Alzheimer's Disease and Related Dementias (ADRD). "
    "Use ONLY the provided context to answer. Do NOT use any outside knowledge. "
    "For multiple-choice questions, respond with ONLY the correct option letter (e.g. A, B, C, D, or E). "
    "For True/False questions, respond with ONLY 'Yes' or 'No'."
)
# SYSTEM_PROMPT = (
#     "You are a clinical expert in ADRD. Your goal is to provide accurate answers based EXCLUSIVELY on the provided retrieved context. "
#     "Do NOT use any outside knowledge. If the answer is not contained in the context, state 'Insufficient evidence'."
#     "\n\nFOLLOW THESE STEPS FOR EACH QUESTION:"
#     "1. EVIDENCE ANALYSIS: Identify the key phrases or facts in the retrieved context that directly support or refute the options/statement."
#     "2. LOGICAL SYNTHESIS: If a question requires multiple conditions (e.g., 'All of the above'), verify that all conditions are met by the retrieved evidence."
#     "3. FINAL OUTPUT:"
#     "   - For Multiple-Choice (MC): Respond with ONLY the single correct letter (A/B/C/D/E)."
#     "   - For True/False (TF): Respond with ONLY 'Yes' or 'No'."
#     "\n\nSTRICT CONSTRAINTS:"
#     "- If you see conflicting information in the retrieved context, prioritize specific clinical guidance or the most explicit directive."
#     "- Ensure your final answer is strictly limited to the required format (e.g., just the letter or Yes/No)."
# )
# SYSTEM_PROMPT = (
#     "You are a clinical expert in ADRD. Answer the question using ONLY the provided context. "
#     "If the context provides information that supports the statement as true, answer 'Yes'. "
#     "If the context indicates the statement is false or provides evidence that contradicts the statement, answer 'No'. "
#     "If the information is completely missing or cannot be inferred from the context, answer 'Insufficient evidence'. "
#     "\n\nSTRICT FORMAT RULES:"
#     "- For Multiple-Choice (MC), output ONLY the single letter (e.g., A, B, C, D, or E)."
#     "- For True/False (TF), output ONLY 'Yes' or 'No'."
#     "- DO NOT explain your reasoning, do not output analysis steps, and do not provide labels."
# )
# SYSTEM_PROMPT = (
#     "You are an expert on Alzheimer's Disease and Related Dementias (ADRD). "
#     "Use ONLY the provided context to answer. Do NOT use any outside knowledge. "
#     "\n\nRULES:"
#     "1. For Multiple-Choice questions, respond with ONLY the correct option letter (e.g., A, B, C, D, or E)."
#     "2. For True/False questions, if the context confirms the statement, respond with ONLY 'Yes'. "
#     "   If the context contradicts the statement or lacks sufficient information, respond with ONLY 'No'."
#     "3. DO NOT include any reasoning, analysis, or explanations."
# )

QUESTION_ONLY_TEMPLATE = "Question:\n{question}"

RAG_USER_TEMPLATE = (
    "Retrieved Context:\n{context}\n\n"
    "Question:\n{question}"
)

@dataclass(frozen=True)
class GenerationConfig:
    max_context_snippets: int = 10
    openai_temperature: float = 0.1
    openai_max_tokens: int = 350
    gemini_max_output_tokens: int = 350


GEN_CONFIG = GenerationConfig()


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    output_prefix: str
    label: str


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt4":          ModelConfig("gpt-4",                                     "answers_gpt4",          "GPT-4"),
    "gpt4_ft":       ModelConfig("ft:gpt-4o-2024-08-06:braincheck::CuWm5MrG", "answers_gpt4_ft",       "GPT-4o Fine-tuned"),
    "gpt5.2":        ModelConfig("gpt-5.2-chat-latest",                        "answers_gpt5.2",        "GPT-5.2"),
    "gemini3_flash": ModelConfig("gemini-3-flash-preview",                     "answers_gemini3_flash", "Gemini 3 Flash"),
    "gemini3_pro":   ModelConfig("gemini-3-pro-preview",                       "answers_gemini3_pro",   "Gemini 3 Pro"),
    "llama3.2":      ModelConfig("llama3.2",                                   "answers_llama3.2",      "Llama-3.2 (Ollama)"),
    "llama3.2_ft":   ModelConfig("llama-3.2-3b-instruct.Q4_K_M.gguf",         "answers_llama3.2_ft",   "Llama-3.2-3B-Instruct-FT (GGUF Q4_K_M)"),
    "llama3.3":      ModelConfig("unsloth/Llama-3.3-70B-Instruct-bnb-4bit",   "answers_llama3.3",      "Llama-3.3-70B-Instruct"),
    "llama3.3_ft":   ModelConfig("unsloth/Llama-3.3-70B-Instruct-bnb-4bit",   "answers_llama3.3_ft",   "Llama-3.3-70B-Instruct-FT (LoRA)"),
}


def format_user_prompt(context: str, question: str) -> str:
    """Build a unified user prompt for generation scripts."""
    if context and context.strip():
        return RAG_USER_TEMPLATE.format(context=context, question=question)
    return QUESTION_ONLY_TEMPLATE.format(question=question)


def build_context_from_passages(
    passages: list[str],
    max_snippets: int = 3,
    scores: list[float] = None,
) -> str:
    """
    Join retrieved passages into a single context string for generation.
    If scores are provided, passages are sorted by score (highest first)
    before the top-max_snippets are selected.
    """
    if not passages:
        return ""
    if scores and len(scores) == len(passages):
        paired = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        passages = [p for _, p in paired]
    return "\n\n".join(
        [f"--- Snippet {i + 1} ---\n{p}" for i, p in enumerate(passages[:max_snippets])]
    )
