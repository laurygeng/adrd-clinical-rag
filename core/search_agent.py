#!/usr/bin/env python3
"""
Search Agent (Free & Medical-Focused: EuropePMC + DuckDuckGo)
Features: Parallel Hybrid Retrieval, Multi-Gap Support, Concurrent LLM Refinement, and Pure Markdown Tracing.
"""

import os
import re
import logging
import requests
import concurrent.futures
from typing import List, Dict, Tuple
from core.llm_engine import LocalLLMClient

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

UA = {"User-Agent": "adrd-medical-rag/2.0"}
logger = logging.getLogger(__name__)

_UNKNOWN_PATTERNS = [
    r"\bi\s*don't\s*know\b", r"\bi\s*do\s*not\s*know\b", r"\bnot\s+enough\s+(?:information|info)\b",
    r"\binsufficient\s+(?:information|info)\b", r"\bcannot\s+determine\b", r"\bcan't\s+determine\b",
    r"\bunable\s+to\s+determine\b", r"\bbased\s+on\s+the\s+provided\s+context\b",
]

def _looks_unknown(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t: return True
    for p in _UNKNOWN_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE): return True
    return False

def clean_search_query_text(text: str, fallback: str = "") -> str:
    raw = str(text or "").replace("\r\n", "\n").strip()
    fb = str(fallback or "").replace("\r\n", "\n").strip()

    if not raw or _looks_unknown(raw):
        raw = fb

    raw = re.sub(r"```.*?```", " ", raw, flags=re.S)
    raw = re.split(r"(?i)\blet me know if you need", raw, maxsplit=1)[0]
    raw = re.sub(r"(?i)^\s*(search\s+query|query|google\s+query|output)\s*:\s*", "", raw).strip()
    raw = re.sub(r"(?i).*?single most important piece of information still missing(?: from the context)?(?: needed to [^:]+)?\s*(?:is|:)\s*", "", raw).strip()
    raw = re.sub(r"(?i).*?missing information(?: from the context)?(?: needed to [^:]+)?\s*(?:is|:)\s*", "", raw).strip()
    raw = re.sub(r"(?i)\bwithout this information.*$", "", raw).strip()

    parts = []
    for line in raw.splitlines():
        line = line.strip().strip("-*•`\"'")
        if not line: continue
        if _looks_unknown(line): continue
        if re.match(r"(?i)^\(note[:\)]", line): continue
        if re.match(r"(?i)^(here('| i)?s|this query|note:|answer:|context:|question:|statement:)\b", line): continue
        parts.append(line)

    cleaned = parts[0] if parts else raw
    cleaned = re.split(r"(?i)\b(?:because|without|therefore|so that|which means|this means)\b", cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip("\"'")
    cleaned = cleaned[:160].strip()

    if len(cleaned) < 3 or _looks_unknown(cleaned):
        cleaned = re.sub(r"\s+", " ", fb).strip().strip("\"'")[:160]

    return cleaned

def _rule_fallback_query(target_info: str, question: str = "") -> str:
    base = clean_search_query_text(target_info, fallback=question)
    if not base:
        base = clean_search_query_text(question, fallback="dementia clinical facts")
    return base[:160]

def _generate_query(client: LocalLLMClient, target_info: str, question: str = "", model: str = "gpt-4o", q_type: str = "MC") -> str:
    if str(q_type).strip().upper() == "QA":
        prompt = (
            "You are a medical research assistant. Given a target missing fact and the original question, "
            "generate ONE highly specific, concise medical search query (3-6 keywords) focused on Alzheimer's Disease, "
            "dementia care guidelines, or clinical facts.\n"
            "CRITICAL: Focus the query on the core medical entity or specific scenario mentioned. Do NOT generalize.\n"
            "Output ONLY the query text, with no quotes, markdown, or preamble.\n\n"
            f"Original Question: {question}\n"
            f"Missing Target Info: {target_info}\nQuery:"
        )
    else:
        prompt = (
            "You are a medical research assistant. Given a target missing fact and the original question, "
            "generate ONE highly specific, concise medical search query (3-6 keywords) focused on Alzheimer's Disease, "
            "dementia care guidelines, or clinical facts.\n"
            "IMPORTANT: Focus the query on the core medical entity or general topic to retrieve standard facts or best practices. "
            "Ignore the specific behavioral claims or rigid constraints from the question, as they might be false premises/myths.\n"
            "Output ONLY the query text, with no quotes, markdown, or preamble.\n\n"
            f"Original Question: {question}\n"
            f"Missing Target Info: {target_info}\nQuery:"
        )

    fallback_query = _rule_fallback_query(target_info=target_info, question=question)
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=40, messages=[{"role": "user", "content": prompt}]
        )
        raw = (r.choices[0].message.content or "").strip()
        if _looks_unknown(raw): return fallback_query
        q = clean_search_query_text(raw, fallback=fallback_query)
        return q if q and not _looks_unknown(q) else fallback_query
    except Exception:
        return fallback_query

def _refine_evidence_text(client: LocalLLMClient, query: str, raw_text: str, model: str = "gpt-4o") -> str:
    if not raw_text or len(raw_text) < 20: return ""
    prompt = (
        f"Extract the specific paragraphs and clinical facts from the text below that directly answer or verify: '{query}'. "
        "Discard unrelated chit-chat or website navigation, but KEEP the full context of the relevant medical facts. Do NOT over-compress.\n\n"
        f"Text:\n{raw_text[:1200]}\n\nExtracted Facts:"
    )
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=250, messages=[{"role": "user", "content": prompt}]
        )
        txt = (r.choices[0].message.content or "").strip()
        if not txt or _looks_unknown(txt): return raw_text[:400]
        return txt
    except Exception:
        return raw_text[:400]

def _europepmc(query: str, n: int = 2) -> List[Dict[str, str]]:
    med_query = query if "dementia" in query.lower() or "alzheimer" in query.lower() else f"{query} dementia"
    params = {"query": med_query, "format": "json", "pageSize": n, "resultType": "core"}
    try:
        d = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params, headers=UA, timeout=10).json()
        out = []
        for x in (d.get("resultList", {}) or {}).get("result", []) or []:
            title = (x.get("title") or "").strip()
            text = x.get("abstractText") or x.get("title") or ""
            text = re.sub(r"<[^>]+>", " ", text).strip()
            src = (x.get("source") or "").strip()
            pid = (x.get("id") or "").strip()
            url = f"https://europepmc.org/article/{src}/{pid}" if src and pid else ""
            if text: out.append({"source": "europepmc", "title": title, "url": url, "text": text[:800]})
        return out
    except Exception:
        return []

def _duckduckgo(query: str, n: int = 2) -> List[Dict[str, str]]:
    if not HAS_DDGS: return []
    out = []
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=n)]
            for res in results:
                body = (res.get("body") or "").strip()
                if not body: continue
                out.append({"source": "duckduckgo", "title": (res.get("title") or "").strip(), "url": (res.get("href") or "").strip(), "text": body[:800]})
    except Exception:
        pass
    return out

def _local_fallback(query: str, retriever) -> List[Dict[str, str]]:
    if retriever is None: return []
    try:
        wp, _, _ = retriever.get_retrieved_passages(query, top_k=4, pre_k=20, window_size=500)
        out = []
        for p in wp[:4]:
            text = (p or "").strip()
            if text: out.append({"source": "local_fallback", "title": "", "url": "", "text": text})
        return out
    except Exception:
        return []

def research(client: LocalLLMClient, target_info: str, question: str = "", retriever=None, q_type: str = "MC", model_name: str = "gpt-4o") -> Tuple[List[Dict[str, str]], str, str]:
    # Start capturing steps for the MD file
    log_lines = ["\n## Completion Retrieval (Web Search) Log\n"]
    logger.info("Search Agent started: Suppressing console debugs, routing details to MD trace.")
    
    # 1. Split Multiple Gaps and Generate Queries Concurrently
    raw_gaps = [g.strip() for g in target_info.split("|") if g.strip()]
    if not raw_gaps: raw_gaps = [target_info]
    
    queries = []
    log_lines.append("### Query Generation")
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(raw_gaps), 3)) as executor:
        future_to_gap = {executor.submit(_generate_query, client, g, question, model_name, q_type): g for g in raw_gaps}
        for future in concurrent.futures.as_completed(future_to_gap):
            original_gap = future_to_gap[future]
            q = future.result()
            if q:
                queries.append(q)
                log_lines.append(f"- **Gap**: `{original_gap}` ➔ **LLM Query**: `{q}`")
            
    queries = list(set(queries))
    if not queries: 
        queries = [_rule_fallback_query(target_info, question)]
        log_lines.append(f"- **Fallback Triggered**: `{queries[0]}`")

    # 2. Parallel Hybrid Search
    # log_lines.append("\n### Search Execution")
    # raw_evidence = []
    # with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    #     search_futures = []
    #     for q in queries:
    #         search_futures.append(executor.submit(_europepmc, q, 2))
    #         if HAS_DDGS: search_futures.append(executor.submit(_duckduckgo, q, 2))
            
    #     for future in concurrent.futures.as_completed(search_futures):
    #         res = future.result()
    #         if res: raw_evidence.extend(res)
    
    # 2. Parallel Hybrid Search
    log_lines.append("\n### Search Execution")
    raw_evidence = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        search_futures = []
        
        # 照护类/生活类词汇库
        care_keywords = ["care", "home", "clothing", "behavior", "agitation", "sleep", "dressing", "eat", "daily", "supervision", "diet", "driving", "financial"]
        
        for q in queries:
            q_lower = q.lower()
            is_care_query = any(k in q_lower for k in care_keywords)
            
            # 核心策略：如果是照护细节题，压制 EuropePMC(学术文献降为1)，增加 DuckDuckGo(通俗网页升为3)
            epmc_limit = 1 if is_care_query else 2
            ddg_limit = 3 if is_care_query else 2
            
            search_futures.append(executor.submit(_europepmc, q, epmc_limit))
            if HAS_DDGS: 
                search_futures.append(executor.submit(_duckduckgo, q, ddg_limit))

    # Deduplicate
    unique_evidence = []
    seen_urls, seen_texts = set(), set()
    for item in raw_evidence:
        url = item.get("url", "")
        text_hash = item.get("text", "")[:100].lower()
        if url and url in seen_urls: continue
        if text_hash and text_hash in seen_texts: continue
        if url: seen_urls.add(url)
        if text_hash: seen_texts.add(text_hash)
        unique_evidence.append(item)

    if not unique_evidence:
        log_lines.append("- **Web Search Yielded 0 Results**. Triggering Local Fallback...")
        for q in queries:
            loc_res = _local_fallback(q, retriever)
            unique_evidence.extend(loc_res)
    else:
        log_lines.append(f"- Retrieved {len(unique_evidence)} unique items from external sources.")

    # 3. Parallel LLM Evidence Refinement
    log_lines.append("\n### LLM Evidence Refinement (Noise Filtering)")
    refined_evidence = []
    
    def _refine_task(idx, item, q_list):
        combined_q = " OR ".join(q_list)
        raw_txt = item.get("text", "")
        source = item.get("source", "unknown")
        title = item.get("title", "No Title")
        url = item.get("url", "No URL")
        refined_txt = _refine_evidence_text(client, combined_q, raw_txt, model=model_name)
        return idx, source, title, url, raw_txt, refined_txt

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        refine_futures = [executor.submit(_refine_task, i, item, queries) for i, item in enumerate(unique_evidence)]
        for future in concurrent.futures.as_completed(refine_futures):
            idx, source, title, url, raw_txt, refined_txt = future.result()
            if refined_txt and not _looks_unknown(refined_txt):
                refined_evidence.append({"source": source, "title": title, "url": url, "text": refined_txt, "idx": idx})
                log_lines.append(f"- ✅ **[KEPT]** Source: `{source}` | URL: {url}")
            elif raw_txt:
                refined_evidence.append({"source": source, "title": title, "url": url, "text": raw_txt[:400], "idx": idx})
                log_lines.append(f"- ⚠️ **[KEPT_RAW]** Source: `{source}` | Reason: LLM refinement failed or bypassed.")
            else:
                log_lines.append(f"- ❌ **[DROPPED]** Source: `{source}` | Reason: Empty or invalid text.")

    # Restore order
    refined_evidence.sort(key=lambda x: x.get("idx", 0))
    for item in refined_evidence: item.pop("idx", None)

    final_ev = refined_evidence if refined_evidence else unique_evidence
    returned_queries = " | ".join(queries)
    log_lines.append(f"\n- **Final Verification**: {len(final_ev)} evidence items sent to Integration Agent.\n")
    
    return final_ev, returned_queries, "\n".join(log_lines)