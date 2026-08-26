#!/usr/bin/env python3
"""
Search Agent
Role: Generates a single targeted web search query based on missing information
and retrieves external evidence using free backends concurrently.

This module returns STRUCTURED evidence items:
    {"source": str, "title": str, "url": str, "text": str}

Backends:
- DuckDuckGo HTML (free) - General & Domain-Specific
- EuropePMC (free) - Keyword optimized
- Optional fallback: local retriever
"""

import re
from html import unescape
from urllib.parse import parse_qs, unquote, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import requests
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

    parts = []
    for line in raw.splitlines():
        line = line.strip().strip("-*•`\"'")
        if not line:
            continue
        if re.match(r"(?i)^\(note[:\)]", line):
            continue
        if re.match(r"(?i)^(here('| i)?s|this query|note:)\b", line):
            continue
        if re.match(r"(?i)^(search\s+query|query|google\s+query|output)\s*:", line):
            line = re.sub(r"(?i)^(search\s+query|query|google\s+query|output)\s*:", "", line).strip()
        parts.append(line)

    cleaned = parts[0] if parts else raw
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip("\"'")
    cleaned = re.sub(r"\s+[\-–—]\s+.*$", "", cleaned).strip()
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
        return clean_search_query_text(target_info, fallback=target_info)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _decode_duckduckgo_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        return "https:" + raw
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if "uddg=" not in raw:
        return raw
    try:
        qs = parse_qs(urlparse(raw).query)
        target = qs.get("uddg", [""])[0]
        return unquote(target) if target else raw
    except Exception:
        return raw


def _duckduckgo_html(query: str, n: int = 6, source_label: str = "duckduckgo") -> List[Dict[str, str]]:
    """DuckDuckGo HTML backend (free). Returns structured evidence items."""
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=UA,
            timeout=10, # Reduced timeout for concurrent execution
        )
        if resp.status_code != 200:
            return []
            
        html = resp.text or ""
        out: List[Dict[str, str]] = []

        for match in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.S):
            start = match.start()
            block = html[start:start + 2500]
            url = _decode_duckduckgo_url(match.group(1))
            title = _strip_html(match.group(2))

            snippet_match = re.search(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', block, flags=re.S)
            if not snippet_match:
                snippet_match = re.search(r'<div[^>]*class="result__snippet"[^>]*>(.*?)</div>', block, flags=re.S)
            text = _strip_html(snippet_match.group(1) if snippet_match else "") or title

            if not text:
                continue
            out.append(
                {
                    "source": source_label,
                    "title": title[:240],
                    "url": url,
                    "text": text[:800],
                }
            )
            if len(out) >= n:
                break

        return out
    except Exception:
        return []


def _europepmc(query: str, n: int = 3) -> List[Dict[str, str]]:
    """EuropePMC backend (free). Returns structured evidence items."""
    # Convert natural language to keyword query for better boolean retrieval
    stop_words = r"\b(what|is|are|the|of|in|on|for|to|with|and|a|an|do|does|how|why)\b"
    kw_query = re.sub(stop_words, "", query, flags=re.IGNORECASE)
    kw_query = re.sub(r"\s+", " ", kw_query).strip()
    
    # Fallback to original query if completely stripped
    if not kw_query:
        kw_query = query
        
    try:
        params = {"query": kw_query, "format": "json", "pageSize": n, "resultType": "core"}
        d = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params=params,
            headers=UA,
            timeout=10, # Reduced timeout for concurrent execution
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


def _dedupe_evidence(evidence: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for ev in evidence or []:
        source = (ev.get("source") or "").strip()
        url = (ev.get("url") or "").strip().lower()
        text = re.sub(r"\s+", " ", (ev.get("text") or "").strip().lower())
        if not text:
            continue
        key = url or text[:240]
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source": source,
                "title": (ev.get("title") or "").strip(),
                "url": (ev.get("url") or "").strip(),
                "text": (ev.get("text") or "").strip()[:800],
            }
        )
    return out


def _looks_biomedical_query(query: str) -> bool:
    q = (query or "").lower()
    medical_terms = [
        "guideline", "diagnosis", "diagnostic", "treatment", "therapy", "prevalence",
        "incidence", "trial", "randomized", "biomarker", "pathophysiology", "symptom",
        "screening", "delirium", "agitation", "medication", "risk", "factor", "dementia",
        "alzheimer", "adrd", "caregiver", "burnout", "cognitive", "impairment"
    ]
    # Reduced threshold to 1 to ensure ADRD queries aren't missed
    return sum(term in q for term in medical_terms) >= 1


def _trusted_domain_bonus(url: str) -> float:
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        host = ""

    if not host:
        return 0.0

    trusted_exact = {
        "alz.org",
        "www.alz.org",
        "alzheimers.gov",
        "www.alzheimers.gov",
        "nia.nih.gov",
        "www.nia.nih.gov",
        "nhs.uk",
        "www.nhs.uk",
        "dementiauk.org",
        "www.dementiauk.org",
        "who.int",
        "www.who.int",
        "cdc.gov",
        "www.cdc.gov",
        "mayoclinic.org",
        "www.mayoclinic.org",
        "my.clevelandclinic.org",
    }
    if host in trusted_exact:
        return 0.12 # Increased bonus for definitive ADRD sources
    if host.endswith(".gov"):
        return 0.08
    if host.endswith(".edu"):
        return 0.05
    if host.endswith(".org"):
        return 0.03
    return 0.0


def _score_evidence(query: str, ev: Dict[str, str], retriever) -> float:
    text = "\n".join(
        part for part in [(ev.get("title") or "").strip(), (ev.get("text") or "").strip()] if part
    )
    score = 0.0
    if retriever is not None and text:
        try:
            score = float(retriever.score_text(query=query, text=text))
        except TypeError:
            try:
                score = float(retriever.score_text(query, text))
            except Exception:
                score = 0.0
        except Exception:
            score = 0.0

    source = (ev.get("source") or "").strip().lower()
    source_bonus = {
        "duckduckgo_domain": 0.10, # Give high priority to domain-restricted search
        "duckduckgo": 0.05,
        "europepmc": 0.06,
        "local_fallback": 0.01,
    }.get(source, 0.0)

    return score + source_bonus + _trusted_domain_bonus((ev.get("url") or "").strip())


def _rank_evidence(query: str, evidence: List[Dict[str, str]], retriever) -> List[Dict[str, str]]:
    deduped = _dedupe_evidence(evidence)
    if not deduped:
        return []

    scored = []
    for ev in deduped:
        item = dict(ev)
        item["_score"] = _score_evidence(query, item, retriever)
        scored.append(item)

    scored.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

    out: List[Dict[str, str]] = []
    per_source = {}
    for ev in scored:
        source = (ev.get("source") or "").strip().lower()
        limit = 4 if source == "duckduckgo_domain" else 3 
        if per_source.get(source, 0) >= limit:
            continue
        per_source[source] = per_source.get(source, 0) + 1
        ev.pop("_score", None)
        out.append(ev)

    return out


def _gather(query: str, retriever) -> List[Dict[str, str]]:
    """Collect evidence for the query across configured backends concurrently."""
    ev: List[Dict[str, str]] = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        
        # 1. Standard Web Search
        futures.append(executor.submit(_duckduckgo_html, query, 5, "duckduckgo"))
        
        # 2. Domain-Specific Search (Highly relevant for ADRD RAG)
        trusted_sites_query = f"{query} (site:alz.org OR site:nia.nih.gov OR site:alzheimers.gov)"
        futures.append(executor.submit(_duckduckgo_html, trusted_sites_query, 4, "duckduckgo_domain"))
        
        # 3. Academic/Medical DB Search
        if _looks_biomedical_query(query):
            futures.append(executor.submit(_europepmc, query, 4))
            
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    ev.extend(result)
            except Exception:
                continue

    if len(ev) < 2:
        ev.extend(_local_fallback(query, retriever))

    return _rank_evidence(query, ev, retriever)


def research(client: OpenAI, target_info: str, retriever=None) -> Tuple[List[Dict[str, str]], str]:
    """
    Execute a single-direction web search.

    Returns:
      (evidence_items, search_query_used)
    """
    query = _generate_query(client, target_info)
    evidence = _gather(query, retriever)
    return evidence, query