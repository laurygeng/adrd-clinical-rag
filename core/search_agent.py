#!/usr/bin/env python3
"""
Search Agent
Role: Generates a single targeted web search query based on missing information
and retrieves external evidence using configured free backends.

This module returns STRUCTURED evidence items:
  {"source": str, "title": str, "url": str, "text": str}

Backends:
- Exa (requires EXA_API_KEY)
- EuropePMC (free)
- Tavily (requires TAVILY_API_KEY)
- Optional fallback: local retriever (if all web backends fail)

Note:
We intentionally keep snippets short to reduce noise and to make Court auditing easier.
"""

import os
import re
import requests
from typing import List, Dict, Tuple
from openai import OpenAI

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


def _generate_query(client: OpenAI, target_info: str, model: str = "gpt-4o-mini") -> str:
    """Generate a single precise search query based on the target missing information."""
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
            messages=[{"role": "user", "content": prompt}],
        )
        q = clean_search_query_text(r.choices[0].message.content or "", fallback=target_info)
        return q if q else clean_search_query_text(target_info, fallback=target_info)
    except Exception:
        # Fallback to raw target_info if query generation fails
        return clean_search_query_text(target_info, fallback=target_info)


def _exa(query: str, n: int = 3) -> List[Dict[str, str]]:
    """Exa backend (requires EXA_API_KEY). Returns structured evidence items."""
    key = os.environ.get("EXA_API_KEY")
    if not key:
        return []

    try:
        payload = {
            "query": query,
            "numResults": n,
            "contents": {"text": {"maxCharacters": 800}},
        }
        headers = {"x-api-key": key, "Content-Type": "application/json", **UA}
        d = requests.post(
            "https://api.exa.ai/search",
            headers=headers,
            json=payload,
            timeout=30,
        ).json()

        out: List[Dict[str, str]] = []
        for x in d.get("results", []) or []:
            text = (x.get("text") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "source": "exa",
                    "title": (x.get("title") or "").strip(),
                    "url": (x.get("url") or "").strip(),
                    "text": text[:800],
                }
            )
        return out
    except Exception:
        return []


def _europepmc(query: str, n: int = 3) -> List[Dict[str, str]]:
    """EuropePMC backend (free). Returns structured evidence items."""
    try:
        params = {"query": query, "format": "json", "pageSize": n, "resultType": "core"}
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

            # Best-effort URL construction
            src = (x.get("source") or "").strip()
            pid = (x.get("id") or "").strip()
            url = ""
            if src and pid:
                url = f"https://europepmc.org/article/{src}/{pid}"

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


def _tavily(query: str, n: int = 4) -> List[Dict[str, str]]:
    """Tavily backend (requires TAVILY_API_KEY). Returns structured evidence items."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []

    try:
        payload = {
            "api_key": key,
            "query": query,
            "search_depth": "advanced",
            "max_results": n,
            "include_answer": True,
        }
        d = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            headers=UA,
            timeout=40,
        ).json()

        out: List[Dict[str, str]] = []

        ans = (d.get("answer") or "").strip()
        if ans:
            out.append(
                {
                    "source": "tavily",
                    "title": "tavily_answer",
                    "url": "",
                    "text": ans[:800],
                }
            )

        for x in d.get("results", []) or []:
            text = (x.get("content") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "source": "tavily",
                    "title": (x.get("title") or "").strip(),
                    "url": (x.get("url") or "").strip(),
                    "text": text[:800],
                }
            )

        return out
    except Exception:
        return []


def _local_fallback(query: str, retriever) -> List[Dict[str, str]]:
    """Optional fallback to local retriever when web backends return nothing."""
    if retriever is None:
        return []
    try:
        wp, _, _ = retriever.get_retrieved_passages(
            query,
            top_k=4,
            pre_k=15,
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


def _gather(query: str, retriever) -> List[Dict[str, str]]:
    """Collect evidence for the query across configured backends."""
    ev: List[Dict[str, str]] = []
    ev.extend(_exa(query, 3))
    ev.extend(_europepmc(query, 3))

    if not ev:
        ev.extend(_tavily(query, 4))

    if not ev:
        ev.extend(_local_fallback(query, retriever))

    return ev


def research(client: OpenAI, target_info: str, retriever=None) -> Tuple[List[Dict[str, str]], str]:
    """
    Execute a single-direction web search.

    Returns:
      (evidence_items, search_query_used)

    evidence_items is a list of dicts with keys:
      source, title, url, text
    """
    query = _generate_query(client, target_info)
    evidence = _gather(query, retriever)
    return evidence, query
