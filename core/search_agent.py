#!/usr/bin/env python3
"""
Search Agent (Free & Medical-Focused: EuropePMC + DuckDuckGo)
Role: Generates a targeted medical search query based on missing info and original question,
and retrieves external evidence using completely free backends (EuropePMC medical database + DuckDuckGo).

This module returns STRUCTURED evidence items:
  {"source": str, "title": str, "url": str, "text": str}
"""

import os
import re
import requests
from typing import List, Dict, Tuple
from openai import OpenAI

# Try importing duckduckgo_search if available
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

UA = {"User-Agent": "adrd-medical-rag/2.0"}


def clean_search_query_text(text: str, fallback: str = "") -> str:
    raw = str(text or "").replace("\r\n", "\n").strip()
    fb = str(fallback or "").replace("\r\n", "\n").strip()

    if not raw:
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
        if not line:
            continue
        if re.match(r"(?i)^\(note[:\)]", line):
            continue
        if re.match(r"(?i)^(here('| i)?s|this query|note:|answer:|context:|question:|statement:)\b", line):
            continue
        parts.append(line)

    cleaned = parts[0] if parts else raw
    cleaned = re.split(r"(?i)\b(?:because|without|therefore|so that|which means|this means)\b", cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip("\"'")
    cleaned = cleaned[:160].strip()

    if len(cleaned) < 3:
        cleaned = re.sub(r"\s+", " ", fb).strip().strip("\"'")[:160]

    return cleaned


def _generate_query(client: OpenAI, target_info: str, question: str = "", model: str = "gpt-4o-mini") -> str:
    """Generate a precise medical search query combining question context and target info."""
    prompt = (
        "You are a medical research assistant. Given a target missing fact and the original question, "
        "generate ONE highly specific, concise medical search query (3-6 keywords) focused on Alzheimer's Disease, "
        "dementia care guidelines, or clinical facts.\n"
        "Output ONLY the query text, with no quotes, markdown, or preamble.\n\n"
        f"Original Question: {question}\n"
        f"Missing Target Info: {target_info}\nQuery:"
    )
    try:
        r = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=40,
            messages=[{"role": "user", "content": prompt}],
        )
        q = clean_search_query_text(r.choices[0].message.content or "", fallback=target_info)
        return q if q else clean_search_query_text(target_info, fallback=target_info)
    except Exception:
        return clean_search_query_text(target_info, fallback=target_info)


def _refine_evidence_text(client: OpenAI, query: str, raw_text: str) -> str:
    """Knowledge Refinement: Extract only the sentences relevant to the medical query, stripping noise."""
    if not raw_text or len(raw_text) < 20:
        return ""
    prompt = (
        f"Extract and summarize ONLY the specific clinical or caregiving facts from the text below that directly answer or verify: '{query}'. "
        "Discard unrelated chit-chat, website navigation, or boilerplate text. Keep it under 3 concise sentences.\n\n"
        f"Text:\n{raw_text[:1000]}\n\nExtracted Facts:"
    )
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return raw_text[:400]


def _europepmc(query: str, n: int = 3) -> List[Dict[str, str]]:
    """EuropePMC backend (Free medical literature database - PubMed abstracts)."""
    try:
        # Append Alzheimer's/dementia context if not present to keep medical focus
        med_query = query if "dementia" in query.lower() or "alzheimer" in query.lower() else f"{query} dementia"
        params = {"query": med_query, "format": "json", "pageSize": n, "resultType": "core"}
        d = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params=params,
            headers=UA,
            timeout=30,
        ).json()

        out: List[Dict[str, str]] = []
        for x in (d.get("resultList", {}) or {}).get("result", []) or []:
            title = (x.get("title") or "").strip()
            text = x.get("abstractText") or x.get("title") or ""
            text = re.sub(r"<[^>]+>", " ", text).strip()

            src = (x.get("source") or "").strip()
            pid = (x.get("id") or "").strip()
            url = f"https://europepmc.org/article/{src}/{pid}" if src and pid else ""

            if text:
                out.append(
                    {
                        "source": "europepmc",
                        "title": title,
                        "url": url,
                        "text": text[:800],
                    }
                )
        return out
    except Exception:
        return []


def _duckduckgo(query: str, n: int = 3) -> List[Dict[str, str]]:
    """DuckDuckGo free search backend (No API key required)."""
    if not HAS_DDGS:
        return []
    out = []
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=n)]
            for res in results:
                body = (res.get("body") or "").strip()
                if not body:
                    continue
                out.append(
                    {
                        "source": "duckduckgo",
                        "title": (res.get("title") or "").strip(),
                        "url": (res.get("href") or "").strip(),
                        "text": body[:800],
                    }
                )
    except Exception:
        pass
    return out


def _local_fallback(query: str, retriever) -> List[Dict[str, str]]:
    """Local retriever fallback if web searches return nothing."""
    if retriever is None:
        return []
    try:
        wp, _, _ = retriever.get_retrieved_passages(
            query,
            top_k=6,
            pre_k=20,
            window_size=500,
        )
        out: List[Dict[str, str]] = []
        for p in wp[:10]:
            text = (p or "").strip()
            if text:
                out.append(
                    {
                        "source": "local_fallback",
                        "title": "",
                        "url": "",
                        "text": text,
                    }
                )
        return out
    except Exception:
        return []


def research(client: OpenAI, target_info: str, question: str = "", retriever=None) -> Tuple[List[Dict[str, str]], str]:
    """
    Execute a free, medical-focused external search (EuropePMC -> DuckDuckGo -> Local Fallback)
    with query contextualization and LLM-based evidence refinement.

    Returns:
      (evidence_items, search_query_used)
    """
    query = _generate_query(client, target_info, question=question)
    
    # 1. Prioritize EuropePMC for authoritative medical literature/guidelines
    ev: List[Dict[str, str]] = []
    ev.extend(_europepmc(query, 3))

    # 2. Fallback to DuckDuckGo general free search if literature yields little
    if not ev:
        ev.extend(_duckduckgo(query, 3))

    # 3. Final fallback to local retrieval
    if not ev:
        ev.extend(_local_fallback(query, retriever))

    # 4. Refine evidence: strip noise and extract pure clinical facts using LLM
    refined_evidence: List[Dict[str, str]] = []
    for item in ev:
        raw_txt = item.get("text", "")
        refined_txt = _refine_evidence_text(client, query, raw_txt)
        if refined_txt:
            refined_evidence.append({
                "source": item.get("source", "web"),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "text": refined_txt,
            })

    return refined_evidence if refined_evidence else ev, query