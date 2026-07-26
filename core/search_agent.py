#!/usr/bin/env python3
"""
Search Agent
Role: Generates a single targeted web search query based on missing information
and retrieves external evidence using configured free backends.
"""

import os
import re
import requests
from openai import OpenAI

UA = {"User-Agent": "adrd-medical-rag/2.0"}

def _generate_query(client: OpenAI, target_info: str, model: str = "gpt-4o-mini") -> str:
    """Generates a single, precise search query based on the missing information."""
    prompt = (
        "Given a piece of missing medical information, a question, or a claim, "
        "output ONE concise web search query to find direct evidence or answers. "
        "Keep it Google-friendly (natural language or 3-6 keywords). "
        "Output ONLY the query text, with no quotes or preamble.\n\n"
        f"Target Info: {target_info}\nQuery:"
    )
    try:
        r = client.chat.completions.create(
            model=model, 
            temperature=0.0, 
            max_tokens=40,
            messages=[{"role": "user", "content": prompt}]
        )
        # Strip unexpected quotes or whitespace
        return (r.choices[0].message.content or "").strip().strip('"\'')
    except Exception:
        return target_info

def _exa(query: str, n: int = 3) -> list:
    key = os.environ.get("EXA_API_KEY")
    if not key: return []
    try:
        payload = {
            "query": query, 
            "numResults": n, 
            "contents": {"text": {"maxCharacters": 600}}
        }
        headers = {"x-api-key": key, "Content-Type": "application/json"}
        d = requests.post("https://api.exa.ai/search", headers=headers, json=payload, timeout=30).json()
        return [(x.get("text") or "").strip() for x in d.get("results", []) if (x.get("text") or "").strip()]
    except Exception:
        return []

def _europepmc(query: str, n: int = 3) -> list:
    try:
        params = {"query": query, "format": "json", "pageSize": n, "resultType": "core"}
        d = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params, headers=UA, timeout=30).json()
        out = []
        for x in d.get("resultList", {}).get("result", []):
            t = x.get("abstractText") or x.get("title") or ""
            t = re.sub(r"<[^>]+>", " ", t).strip()
            if t: out.append(t[:600])
        return out
    except Exception:
        return []

def _tavily(q: str, key: str, n: int = 4) -> list:
    try:
        payload = {"api_key": key, "query": q, "search_depth": "advanced", "max_results": n, "include_answer": True}
        d = requests.post("https://api.tavily.com/search", json=payload, timeout=40).json()
        return [s for s in [(d.get("answer") or "")] + [x.get("content", "") for x in d.get("results", [])] if s.strip()]
    except Exception:
        return []

def _gather(query: str, retriever) -> list:
    """Collect evidence for the query across the configured free backends."""
    ev = _exa(query, 3) + _europepmc(query, 3)
    if not ev and os.environ.get("TAVILY_API_KEY"):
        ev = _tavily(query, os.environ["TAVILY_API_KEY"])
        
    # Optional fallback to local retriever if all web backends fail
    if not ev and retriever is not None:
        try:
            wp, _, _ = retriever.get_retrieved_passages(query, top_k=4, pre_k=15, window_size=500)
            ev = wp[:10]
        except Exception:
            ev = []
    return ev

def research(client: OpenAI, target_info: str, retriever=None) -> tuple:
    """
    Executes a single-direction web search.
    Returns: (list_of_snippets, search_query_used)
    """
    query = _generate_query(client, target_info)
    snippets = _gather(query, retriever)
    return snippets, query