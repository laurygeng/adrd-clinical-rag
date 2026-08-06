def extract_and_cite_facts(query, contexts, model="gpt-4o"):
    """
    Leaf Module: Extract any relevant facts from contexts.
    """
    client = get_openai_client()
    # Strictly require JSON object with 'extracted_facts' key
    system_prompt = (
        "You are a medical information extractor. From the provided Context array, extract ANY factual statements related to the entities, concepts, or options mentioned in the user's Query. "
        "Even if the fact only partially answers the query, you MUST extract it. Do not infer, keep the original wording. "
        "You MUST return a JSON object with a single key 'extracted_facts' containing an array of the facts. "
        "Each item in the array must be an object with 'fact' (string) and 'source_index' (integer). "
        "If absolutely no related information is found, return {\"extracted_facts\": []}."
    )
    indexed_contexts = {f"[{i}]": text for i, text in enumerate(contexts)}
    user_content = {"Query": query, "Context": indexed_contexts}
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
        return content if content else '{"extracted_facts": []}'
    except Exception as e:
        print(f"Extraction API failed: {e}")
        return '{"extracted_facts": []}'
import openai
import os

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
    return client


def evaluate_context(original_question, retrieved_contexts, model="gpt-4o"):
    """
    Evaluate whether the retrieved_contexts are sufficient to answer the original_question.
    Returns a dict: {"status": "answerable"/"unanswerable", "missing_information": ""/details}
    """
    client = get_openai_client()
    system_prompt = (
        "You are a strict information evaluation expert. Judge whether the provided retrieved_contexts are sufficient to answer the original_question. "
        "Do NOT answer based on your own knowledge, only use the given contexts. "
        "Output in JSON format. "
        "If the contexts are sufficient, output {\"status\": \"answerable\", \"missing_information\": \"\"}. "
        "If the information is insufficient, output {\"status\": \"unanswerable\", \"missing_information\": \"Summarize what specific information is missing\"}."
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
        return content if content else '{"status": "answerable", "missing_information": ""}'
    except Exception as e:
        print(f"Evaluate Context API failed: {e}")
        return '{"status": "answerable", "missing_information": ""}'


def rewrite_tf_query(statement, model="gpt-4o"):
    """
    优化1+2：将 TF 陈述句重写为疑问句，并拆解复合声明为子 Claim 查询。
    返回 {"queries": ["疑问句1", "子声明查询2", ...]}（2-4 条）
    """
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


def evaluate_tf_evidence(statement, retrieved_contexts, model="gpt-4o"):
    """
    优化4：Evidence-first TF 判断。
    要求模型先从 context 中提取关键证据，再判断陈述真假。
    若证据不足，返回 insufficient 触发下一轮检索而非堆砌噪音。

    返回:
        {
          "verdict": "True" | "False" | "insufficient",
          "evidence": "引用的核心证据原文",
          "missing": "缺少什么信息（仅 insufficient 时有值）"
        }
    """
    client = get_openai_client()
    system_prompt = (
        "You are a strict medical fact-checker for ADRD (Alzheimer's Disease and Related Dementias). "
        "You will be given a statement and a list of retrieved context passages. "
        "Your task:\n"
        "Step 1 — Scan the passages for any sentence that DIRECTLY supports or refutes "
        "the specific factual claim in the statement (e.g. a statistic, a definition, a comparison).\n"
        "Step 2 — If you find a directly relevant sentence, output it verbatim as 'evidence' "
        "and give a 'verdict' of 'True' or 'False'.\n"
        "Step 3 — If NO passage contains a sentence that directly addresses the core claim, "
        "output verdict='insufficient' and describe in 'missing' what specific fact is needed.\n"
        "CRITICAL: Do NOT infer, extrapolate, or use outside knowledge. "
        "Output JSON: {\"verdict\": \"True|False|insufficient\", \"evidence\": \"...\", \"missing\": \"...\"}"
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
        return content if content else '{"verdict": "insufficient", "evidence": "", "missing": "API returned empty"}'
    except Exception as e:
        print(f"evaluate_tf_evidence API failed: {e}")
        return '{"verdict": "insufficient", "evidence": "", "missing": "API error"}'


def decompose_mc_options(stem, options_dict, model="gpt-4o"):
    """
    MC 选项拆解：为每个选项独立生成一条针对性的事实核查检索 Query。
    输入: stem (题干), options_dict (e.g. {"A": "...", "B": "...", "C": "...", "D": "..."})
    返回: {"option_queries": {"A": "query_a", "B": "query_b", ...}}
    """
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
        "Output JSON: {\"option_queries\": {\"A\": \"query_a\", \"B\": \"query_b\", \"C\": \"query_c\", \"D\": \"query_d\"}}"
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


def refine_passages_for_mc(stem, option_queries_map, contexts, model="gpt-4o-mini"):
    """
    CRAG/MIGRES Knowledge Refinement：将检索到的长段落提炼成针对选项的要点子弹头，
    过滤掉所有与选项判断无关的背景噪声。
    使用 gpt-4o-mini 以控制 cost。
    返回: {"refined": ["[A] 要点 bullet1", "[B] 要点 bullet2", ...]} （最多 8 条）
    """
    client = get_openai_client()
    system_prompt = (
        "You are a medical knowledge distiller for ADRD (Alzheimer's Disease and Related Dementias). "
        "Given a multiple-choice question stem and retrieved passages, perform KNOWLEDGE REFINEMENT:\n"
        "Step 1: Filter out generic background noise and keep ONLY the passages that contain evidence "
        "to verify or refute any of the specific options.\n"
        "Step 2: Do NOT aggressively summarize. Preserve the full original sentences and the clinical "
        "reasoning context (the 'why' and 'how') from the source text.\n"
        "Step 3: Group the relevant text by option. Format as: '[Option A Evidence]: <full relevant text>...'\n"
        "Output JSON: {\"refined\": [\"[Option A Evidence]: ...\", \"[Option B Evidence]: ...\"]}."
    )
    option_str = "; ".join([f"{k}: {v}" for k, v in option_queries_map.items()]) if isinstance(option_queries_map, dict) else str(option_queries_map)
    context_str = "\n\n".join([f"[Passage {i}]: {p}" for i, p in enumerate(contexts)])
    user_content = f"Stem: {stem}\nOption queries: {option_str}\n\nPassages:\n{context_str}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content[:6000]}  # cap to avoid token overflow
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return content if content else '{"refined": []}'
    except Exception as e:
        print(f"refine_passages_for_mc API failed: {e}")
        return '{"refined": []}'


def generate_gap_claim(question, contexts, q_type="MC", model="gpt-4o"):
    """
    Identify-then-Verify (Gap Claim 生成器)：
    生成一个精确、可证伪的缺口宣言（Gap Claim）和针对性检索子句。
    替代模糊的自由发挥 missing_info，强制输出可执行的精确断言。
    返回: {"gap_claim": "Missing: specific fact about X...", "targeted_query": "\u77ed称精确检索词"}
    """
    client = get_openai_client()
    system_prompt = (
        "You are a retrieval gap analyzer for a medical QA system about ADRD "
        "(Alzheimer's Disease and Related Dementias). "
        "Given a question and the currently retrieved contexts, identify the SINGLE most critical "
        "missing piece of information that prevents answering the question confidently.\n"
        "Output a PRECISE, FALSIFIABLE gap claim \u2014 not a vague 'more information needed' statement.\n"
        "BAD: 'Need more context about dementia care'\n"
        "GOOD: 'Missing: specific recommended technique for redirecting a patient who refuses bathing'\n\n"
        "Also output a short targeted retrieval query (\u226410 words) that would most likely surface the missing fact.\n"
        "Output JSON: {\"gap_claim\": \"...\", \"targeted_query\": \"...\"}"
    )
    context_str = "\n".join([f"[{i}] {c[:200]}" for i, c in enumerate(contexts[:10])])
    user_content = f"Question ({q_type}): {question}\n\nCurrent contexts:\n{context_str}"
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
        return content if content else '{"gap_claim": "", "targeted_query": ""}'
    except Exception as e:
        print(f"generate_gap_claim API failed: {e}")
        return '{"gap_claim": "", "targeted_query": ""}'


def generate_queries(original_question, missing_information, model="gpt-4o"):
    """
    Generate 1-3 simple sub-queries based on the original question and missing information.
    Returns a dict: {"queries": ["sub-query1", ...]}
    """
    client = get_openai_client()
    system_prompt = (
        "Based on the original question and missing information, generate 1 to 3 minimal search queries (single-hop) for the local search engine. "
        "Output in JSON format like this: {\"queries\": [\"query1\", \"query2\"]}."
    )
    user_content = {
        "original_question": original_question,
        "missing_information": missing_information
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
        return content if content else '{"queries": []}'
    except Exception as e:
        print(f"Generate Queries API failed: {e}")
        return '{"queries": []}'
