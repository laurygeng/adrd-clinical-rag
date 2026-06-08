import openai
import os
from rag_config import config

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
    return client

def evaluate_context(original_question, retrieved_contexts, model=None):
    """
    Evaluate whether the retrieved_contexts are sufficient to answer the original_question.
    """
    if model is None: model = config.llm_eval_model
    client = get_openai_client()
    system_prompt = (
        "You are a strict information evaluation expert. Judge whether the provided retrieved_contexts are sufficient to answer the original_question. "
        "Do NOT answer based on your own knowledge, only use the given contexts. "
        "Output in JSON format. "
        "Provide a 'reasoning' step to explain your decision. "
        "If the contexts are sufficient, output {\"status\": \"answerable\", \"missing_information\": \"\", \"reasoning\": \"...\"}. "
        "If the information is insufficient, output {\"status\": \"unanswerable\", \"missing_information\": \"Summarize what specific information is missing\", \"reasoning\": \"...\"}."
    )
    user_content = {
        "original_question": original_question,
        "retrieved_contexts": retrieved_contexts
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_content)}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return content if content else '{"status": "answerable", "missing_information": "", "reasoning": ""}'
    except Exception as e:
        print(f"Evaluate Context API failed: {e}")
        return '{"status": "answerable", "missing_information": "", "reasoning": ""}'

def rewrite_tf_query(statement, model=None):
    """
    Rewrite a True/False statement into an interrogative query and decompose compound claims into sub-queries.
    """
    if model is None: model = config.llm_rewrite_model
    client = get_openai_client()
    system_prompt = (
        "You are a fact-checking query optimizer for a medical retrieval system. "
        "Given a True/False statement about ADRD (Alzheimer's Disease and Related Dementias), "
        "do TWO things:\n"
        "1. Rewrite it as 1-2 interrogative (question) forms that will help retrieve documents "
        "containing the key facts needed to VERIFY OR REFUTE the statement. "
        "Focus on the core measurable claim (numbers, proportions, comparisons, definitions).\n"
        "2. If the statement contains multiple sub-claims (e.g. condition A AND condition B), "
        "decompose it into separate atomic queries, one per sub-claim.\n"
        "Output JSON: {\"queries\": [\"query1\", \"query2\", ...]}"
        " — 2 to 4 queries total, NO duplicates."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Statement: {statement}"}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return content if content else '{"queries": []}'
    except Exception as e:
        print(f"rewrite_tf_query API failed: {e}")
        return '{"queries": []}'

def evaluate_tf_evidence(statement, retrieved_contexts, model=None):
    """
    TF evaluator with Chain of Thought (CoT) and clinical synonym mapping mechanisms.
    """
    if model is None: model = config.llm_eval_model
    client = get_openai_client()
    system_prompt = (
        "You are an expert clinical evidence evaluator for ADRD (Alzheimer's Disease and Related Dementias). "
        "Your task is to determine if the given statement is True or False based STRICTLY on the provided retrieved passages.\n\n"
        "Follow these steps:\n"
        "1. Read the statement and all passages carefully.\n"
        "2. Identify if the passages contain explicit evidence OR clinically synonymous concepts that logically support or refute the statement. "
        "Basic logical synthesis across multiple passages is allowed, but strictly DO NOT hallucinate or use outside knowledge.\n"
        "3. Output a JSON object with three keys:\n"
        "   - 'reasoning': A brief step-by-step explanation of how the text connects to the statement.\n"
        "   - 'evidence': The exact quotes from the text that support your reasoning.\n"
        "   - 'verdict': Must be exactly 'True', 'False', or 'insufficient'. Use 'insufficient' ONLY if the topic is entirely missing or unanswerable from the text."
    )
    context_str = "\n\n".join([f"[{i}] {p}" for i, p in enumerate(retrieved_contexts)])
    user_content = f"Statement: {statement}\n\nRetrieved Passages:\n{context_str}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return content if content else '{"verdict": "insufficient", "evidence": "", "reasoning": "API returned empty"}'
    except Exception as e:
        print(f"evaluate_tf_evidence API failed: {e}")
        return '{"verdict": "insufficient", "evidence": "", "reasoning": "API error"}'

def decompose_mc_options(stem, options_dict, model=None):
    """
    Multiple Choice option decomposition: Generate a targeted fact-checking retrieval query independently for each option.
    """
    if model is None: model = config.llm_rewrite_model
    client = get_openai_client()
    system_prompt = (
        "You are a medical information retrieval expert specializing in ADRD "
        "(Alzheimer's Disease and Related Dementias). "
        "Given a multiple-choice question stem and its options, generate ONE focused factual "
        "search query PER OPTION that would retrieve documents directly verifying or refuting "
        "that specific option's claim. "
        "Each query must:\n"
        "  1. Combine the key concept from the stem with the specific factual claim in that option.\n"
        "  2. Be phrased as a factual keyword/phrase search (≤15 words), NOT a full sentence.\n"
        "  3. Avoid repeating information that is identical across options.\n"
        "Output JSON: A dictionary where keys are the exact option letters provided (e.g., \"A\", \"B\", \"C\", \"D\", \"E\") and values are the generated queries. "
        "Example: {\"option_queries\": {\"A\": \"query_a\", \"B\": \"query_b\", \"C\": \"query_c\", \"E\": \"query_e\"}}"
    )
    options_str = "\n".join([f"{k}. {v}" for k, v in options_dict.items()])
    user_content = f"Stem: {stem}\nOptions:\n{options_str}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return content if content else '{"option_queries": {}}'
    except Exception as e:
        print(f"decompose_mc_options API failed: {e}")
        return '{"option_queries": {}}'