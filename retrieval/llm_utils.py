import openai
import os
import time
from rag_config import config

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
    return client

def _chat_with_retry(client, *, max_retries=3, base_delay=1.5, **kwargs):
    """Call chat.completions.create with exponential backoff on transient errors.

    Raises the last exception if all attempts fail, so callers keep their own
    try/except (and their safe defaults) for the terminal failure case.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_err

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
        response = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_content)}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        # Fail "unanswerable" (not "answerable") so an API/empty failure does NOT
        # masquerade as sufficient context and silently skip the web fallback.
        return content if content else '{"status": "unanswerable", "missing_information": "Local evaluation returned empty (API issue); attempt external retrieval.", "reasoning": "API returned empty"}'
    except Exception as e:
        print(f"Evaluate Context API failed: {e}")
        return '{"status": "unanswerable", "missing_information": "Local evaluation failed (API error); attempt external retrieval.", "reasoning": "API error"}'

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
        response = _chat_with_retry(
            client,
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

def generate_draft_query(question, q_type="TF", options_dict=None, model=None):
    """FLARE/HyDE-style draft-then-retrieve: generate a short hypothetical source-style
    sentence that would CONTAIN the answer, to use as an extra retrieval query. This
    matches the form of source documents better than the generic question, surfacing
    specific facts (numbers, durations, definitions) the question-query misses."""
    if model is None: model = config.llm_rewrite_model
    client = get_openai_client()
    if q_type == "TF":
        sys_p = (
            "You are a retrieval query generator for an ADRD medical/caregiving system. "
            "Given a True/False statement, write 1-2 concise DECLARATIVE sentences as they "
            "would plausibly appear in an authoritative source document stating the relevant "
            "fact (include the specific number/duration/definition if the statement has one). "
            "Do NOT judge true/false — just write the factual source-style sentence(s) to search with. "
            "Output JSON: {\"queries\": [\"sentence1\", \"sentence2\"]}"
        )
        user_c = f"Statement: {question}"
    else:
        sys_p = (
            "You are a retrieval query generator for an ADRD medical/caregiving system. "
            "Given a multiple-choice question, write 1-2 concise DECLARATIVE sentences as they "
            "would appear in an authoritative source document stating the key fact that "
            "discriminates the correct option (include specific numbers/durations if relevant). "
            "Output JSON: {\"queries\": [\"sentence1\", \"sentence2\"]}"
        )
        opts = "\n".join(f"{k}. {v}" for k, v in (options_dict or {}).items())
        user_c = f"Question: {question}\nOptions:\n{opts}"
    try:
        response = _chat_with_retry(
            client, model=model,
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_c}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return content if content else '{"queries": []}'
    except Exception as e:
        print(f"generate_draft_query API failed: {e}")
        return '{"queries": []}'

def generate_gap_query(question, missing_info="", model=None):
    """Turn a question (and the evaluator's 'what's missing') into ONE short, targeted
    local-search phrase naming the specific fact needed — for gap-guided re-retrieval."""
    if model is None: model = getattr(config, "llm_gap_model", "gpt-4o-mini")
    client = get_openai_client()
    sysp = (
        "Given a question (and optionally what information is missing), write ONE short search "
        "phrase (<= 12 words) naming the SPECIFIC fact needed to answer it — a number, definition, "
        "recommendation, or discriminating detail. Output ONLY the phrase, no punctuation."
    )
    user = f"Question: {question}"
    if missing_info:
        user += f"\nMissing: {missing_info}"
    try:
        r = _chat_with_retry(client, model=model, max_tokens=40, temperature=0.0,
                             messages=[{"role": "system", "content": sysp},
                                       {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"generate_gap_query failed: {e}")
        return ""

def agentic_search_queries(question, retrieved_snippets, missing_info, q_type="TF", round_idx=0, model=None):
    """Agentic web-search step: reflect on what has ALREADY been retrieved and what is
    still missing, then issue refined/diverse NEXT search queries targeting that gap.
    Round 0 = obvious phrasings; later rounds must try DIFFERENT angles (synonyms,
    broader/narrower terms, related clinical concepts, authoritative-source phrasings)
    so we don't keep fetching the same content. Returns JSON {queries, still_missing}."""
    if model is None: model = config.llm_rewrite_model
    client = get_openai_client()
    # keep the agent's view of "already have" short
    seen = "\n".join(f"- {s[:200]}" for s in (retrieved_snippets or [])[:8]) or "(nothing retrieved yet)"
    angle = (
        "These are the FIRST queries — use the most direct phrasing of the missing fact."
        if round_idx == 0 else
        f"This is round {round_idx+1}. The previous queries did NOT surface the missing fact. "
        "Try DISTINCTLY DIFFERENT angles: clinical synonyms, broader or narrower terms, the "
        "specific number/definition phrased as a source sentence, guideline/authoritative wording, "
        "or a related concept that would co-occur with the fact. Do NOT repeat earlier phrasings."
    )
    system_prompt = (
        "You are a medical research agent for ADRD (Alzheimer's and related dementias) filling a "
        "knowledge gap from the web. You are given the QUESTION, the snippets ALREADY retrieved, and "
        "what is still MISSING. Reflect, then generate 2-3 web search queries that would find the "
        "missing fact. Keep queries Google-friendly (natural language or 3-6 keywords), NO Boolean "
        "operators or quotes. " + angle + "\n"
        "Output JSON: {\"queries\": [\"q1\", \"q2\"], \"still_missing\": \"<one line>\"}"
    )
    kind = "True/False statement" if q_type == "TF" else "Multiple-choice question"
    user_content = (
        f"{kind}: {question}\n\n"
        f"Already retrieved:\n{seen}\n\n"
        f"Still missing: {missing_info or 'the specific discriminating fact needed to answer'}"
    )
    try:
        response = _chat_with_retry(
            client, model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_content}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return content if content else '{"queries": []}'
    except Exception as e:
        print(f"agentic_search_queries API failed: {e}")
        return '{"queries": []}'

def evaluate_tf_evidence(statement, retrieved_contexts, model=None):
    """
    TF evaluator with Chain of Thought (CoT) and clinical synonym mapping mechanisms.
    Now supports extracting 'missing_information' for downstream web fallback.
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
        "3. Output a JSON object with five keys:\n"
        "   - 'reasoning': A brief step-by-step explanation of how the text connects to the statement.\n"
        "   - 'evidence': The exact quotes from the text that support your reasoning.\n"
        "   - 'verdict': Must be exactly 'True', 'False', or 'insufficient'. Use 'insufficient' ONLY if the topic is entirely missing or unanswerable from the text.\n"
        "   - 'confidence': Must be exactly 'high', 'medium', or 'low'. Use 'high' ONLY when the passages contain explicit, direct evidence that settles the statement; "
        "'medium' when the verdict rests on inference or partial evidence; 'low' when evidence is weak, indirect, or barely related.\n"
        "   - 'missing_information': If verdict is 'insufficient' OR confidence is not 'high', state exactly what specific additional factual evidence would confirm or refute the claim. Otherwise leave empty."
    )
    context_str = "\n\n".join([f"[{i}] {p}" for i, p in enumerate(retrieved_contexts)])
    user_content = f"Statement: {statement}\n\nRetrieved Passages:\n{context_str}"
    try:
        response = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return content if content else '{"verdict": "insufficient", "confidence": "low", "evidence": "", "reasoning": "API returned empty", "missing_information": "API Failure"}'
    except Exception as e:
        print(f"evaluate_tf_evidence API failed: {e}")
        return '{"verdict": "insufficient", "confidence": "low", "evidence": "", "reasoning": "API error", "missing_information": "API Error"}'

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
        response = _chat_with_retry(
            client,
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

def generate_web_queries_from_missing_info(original_question, missing_information, q_type="MC", model=None):
    """
    Generate targeted web search queries based on missing info.
    Optimized for general web search engines (like DuckDuckGo) without strict Boolean operators.
    """
    if not missing_information:
        return '{"queries": []}'
        
    if model is None: model = config.llm_rewrite_model
    client = get_openai_client()
    
    system_prompt = (
        "You are an expert search query generator for medical RAG systems specializing in Alzheimer's Disease and Related Dementias (ADRD). "
        "Your goal is to generate search queries to retrieve missing factual information from a general web search engine (like Google or DuckDuckGo).\n\n"
    )
    
    if q_type == "TF":
        system_prompt += (
            "The original query is a True/False claim. Local retrieval returned 'insufficient'.\n"
            "STRATEGY:\n"
            "1. DECOMPOSE the statement into 1-2 atomic clinical claims based on the missing information.\n"
            "2. Generate 1-2 broad, Google-friendly search queries (natural language or 3-4 loose keywords) targeting the missing evidence.\n"
            "3. Do NOT use Boolean operators (AND/OR) or quotes. Keep it simple for a generic web crawler to find medical articles.\n"
        )
    else:
        system_prompt += (
            "The original query is a Multiple Choice question. Local retrieval lacked specific facts.\n"
            "STRATEGY:\n"
            "1. Analyze the exact 'missing information' identified.\n"
            "2. Generate 1-2 simple, Google-friendly keyword queries targeting the missing facts.\n"
            "3. Do NOT use Boolean operators (AND/OR) or quotes. Use simple descriptive keywords that a standard search engine can easily match.\n"
        )
        
    system_prompt += "Output MUST be strictly JSON format: {\"queries\": [\"query1\", \"query2\"]}"
    
    user_content = f"Original Query: {original_question}\nMissing Information: {missing_information}"
    
    try:
        response = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return content if content else '{"queries": []}'
    except Exception as e:
        print(f"generate_web_queries_from_missing_info API failed: {e}")
        return '{"queries": []}'