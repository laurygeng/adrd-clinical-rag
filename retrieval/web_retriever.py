#!/usr/bin/env python3
"""
web_retriever.py
Supplementary web-based retrieval for ADRD questions when the local
knowledge base is insufficient.

Sources (all free / no paid key required for basic usage):
  1. Wikipedia REST API      – ADRD disease facts, pathology, classifications
  2. PubMed E-utilities API  – Peer-reviewed biomedical abstracts (NCBI, free)
  3. Semantic Scholar API    – Academic paper abstracts (free, no key needed)
  4. DuckDuckGo Search       - Domain-restricted web search (free, no key needed)
  5. Tavily Search API       – General web search (Disabled by default)

Usage:
    from web_retriever import web_augment
    passages, sources, scores = web_augment(query, nli_model, cross_encoder, threshold=0.1)
"""

import os
import re
import logging
import requests
import time
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Optional DuckDuckGo integration (duckduckgo-search)
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None


# ──────────────────────────────────────────
# WIKIPEDIA
# ──────────────────────────────────────────

# ADRD-relevant Wikipedia pages to prioritize in title-based search
ADRD_WIKI_TITLES = [
    "Alzheimer's disease",
    "Dementia",
    "Lewy body dementia",
    "Frontotemporal dementia",
    "Vascular dementia",
    "Mild cognitive impairment",
    "Caregiver",
    "Sundowning",
    "Wandering (dementia)",
]

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_HEADERS = {
    "User-Agent": "ADRD-RAG-Research/1.0 (academic research; contact: adrd-rag@research.local)"
}


def _wiki_search_titles(query: str, limit: int = 3) -> list[str]:
    """Return page titles from Wikipedia full-text search."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "srnamespace": 0,
        "format": "json",
    }
    try:
        resp = requests.get(_WIKI_API, params=params, headers=_WIKI_HEADERS, timeout=10)
        data = resp.json()
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return []


def _wiki_get_extract(title: str, sentences: int = 10) -> Optional[str]:
    """Fetch a plain-text extract of a Wikipedia article."""
    params = {
        "action": "query",
        "prop": "extracts",
        "exsentences": sentences,
        "explaintext": True,
        "exsectionformat": "plain",
        "titles": title,
        "format": "json",
    }
    try:
        resp = requests.get(_WIKI_API, params=params, headers=_WIKI_HEADERS, timeout=10)
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "").strip()
            if extract and not extract.startswith("REDIRECT"):
                return extract
    except Exception as e:
        logger.warning(f"Wikipedia extract failed for '{title}': {e}")
    return None


def _split_into_chunks(text: str, max_chars: int = 1200) -> list[str]:
    """
    Split text into chunks of <= max_chars characters, breaking at sentence
    boundaries where possible. Increased to 1200 to preserve medical context.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 30]


def search_wikipedia(query: str, max_pages: int = 3, chunk_chars: int = 1200) -> list[dict]:
    """
    Search Wikipedia and return a list of text chunks with source labels.
    Returns: [{"text": str, "source": str}, ...]
    """
    titles = _wiki_search_titles(query, limit=max_pages)
    results = []
    for title in titles:
        extract = _wiki_get_extract(title, sentences=15)
        if not extract:
            continue
        for chunk in _split_into_chunks(extract, chunk_chars):
            results.append({
                "text": chunk,
                "source": f"Wikipedia:{title}",
            })
        time.sleep(0.1)   # Be polite to the API
    return results


# ──────────────────────────────────────────
# TAVILY (Temporarily disabled - Uncomment if needed)
# ──────────────────────────────────────────

# def search_tavily(query: str, max_results: int = 5, chunk_chars: int = 1200) -> list[dict]:
#     """
#     Search the web using the Tavily API (free tier).
#     Requires the environment variable TAVILY_API_KEY to be set.
#     """
#     pass


# ──────────────────────────────────────────
# ALZ.ORG (Alzheimer's Association)
# ──────────────────────────────────────────

_ALZ_SEARCH = "https://www.alz.org/search"
_ALZ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ADRD-RAG-Research/1.0; academic use)",
    "Accept": "text/html,application/xhtml+xml",
}


def search_alz_org(query: str, max_results: int = 5, chunk_chars: int = 1200) -> list[dict]:
    """
    Search alz.org and scrape result snippets. No API key required.
    Returns: [{"text": str, "source": str}, ...]
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed. Run: pip install beautifulsoup4")
        return []
    try:
        resp = requests.get(
            _ALZ_SEARCH,
            params={"q": query},
            headers=_ALZ_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logger.warning(f"alz.org search failed: {e}")
        return []

    results = []
    # Grab all <p>, <li>, <div> text blocks that are substantial
    for tag in soup.find_all(["p", "li", "div"], limit=120):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) < 60 or len(text) > 2000:
            continue
        # Filter for relevance: must mention dementia/alzheimer/caregiver
        lower = text.lower()
        if not any(kw in lower for kw in ("alzheimer", "dementia", "caregiver", "memory", "cognitive")):
            continue
        for chunk in _split_into_chunks(text, chunk_chars):
            results.append({"text": chunk, "source": "alz.org"})
        if len(results) >= max_results:
            break
    logger.info(f"alz.org scraped {len(results)} chunks")
    return results


# ──────────────────────────────────────────
# NIA.NIH.GOV (National Institute on Aging)
# ──────────────────────────────────────────

_NIA_SEARCH = "https://www.nia.nih.gov/search"
_NIA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ADRD-RAG-Research/1.0; academic use)",
    "Accept": "text/html,application/xhtml+xml",
}


def search_nia_nih(query: str, max_results: int = 5, chunk_chars: int = 1200) -> list[dict]:
    """
    Search nia.nih.gov and scrape result snippets. No API key required.
    Returns: [{"text": str, "source": str}, ...]
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed. Run: pip install beautifulsoup4")
        return []
    try:
        resp = requests.get(
            _NIA_SEARCH,
            params={"query": query},
            headers=_NIA_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logger.warning(f"NIA search failed: {e}")
        return []

    results = []
    for tag in soup.find_all(["p", "li", "div"], limit=120):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) < 60 or len(text) > 2000:
            continue
        lower = text.lower()
        if not any(kw in lower for kw in ("alzheimer", "dementia", "aging", "cognitive", "caregiver")):
            continue
        for chunk in _split_into_chunks(text, chunk_chars):
            results.append({"text": chunk, "source": "nia.nih.gov"})
        if len(results) >= max_results:
            break
    logger.info(f"NIA scraped {len(results)} chunks")
    return results


# ──────────────────────────────────────────
# DUCKDUCKGO & PUBMED
# ──────────────────────────────────────────

def search_duckduckgo_domain(query: str, domain: str = "caregiving", max_results: int = 5, chunk_chars: int = 1200) -> list[dict]:
    """
    Use DuckDuckGo (DDGS) as a domain-restricted site search to fetch authoritative snippets.
    Falls back to empty list if DDGS not available.
    """
    if DDGS is None:
        logger.info("DDGS not available; duckduckgo-search not installed")
        return []

    if domain == "caregiving":
        site_clause = "site:alz.org OR site:nia.nih.gov OR site:caregiver.org"
    else:
        site_clause = "site:merckmanuals.com OR site:mayoclinic.org OR site:ncbi.nlm.nih.gov"

    domain_query = f"{query} {site_clause}"
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(domain_query, max_results=max_results):
                body = r.get("body", "") or r.get("snippet", "") or ""
                href = r.get("href", r.get("url", "unknown"))
                if not body:
                    continue
                for chunk in _split_into_chunks(body, chunk_chars):
                    results.append({"text": chunk, "source": f"Web:{href}"})
                time.sleep(0.2)
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
    
    time.sleep(0.5) # Polite delay
    return results

_PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_PUBMED_HEADERS = {
    "User-Agent": "ADRD-RAG-Research/1.0 (academic research; contact: adrd-rag@research.local)"
}
# Append ADRD mesh terms to every query for domain focus
_PUBMED_SUFFIX = " AND (dementia[MeSH] OR Alzheimer[MeSH] OR caregiver[MeSH])"


def search_pubmed(query: str, max_results: int = 5, chunk_chars: int = 1200) -> list[dict]:
    """
    Search PubMed via NCBI E-utilities and return abstract chunks.
    Completely free – no API key required (rate-limited to 3 req/s without key).
    Returns: [{"text": str, "source": str}, ...]
    """
    # Step 1: esearch – get PMIDs
    search_params = {
        "db": "pubmed",
        "term": query + _PUBMED_SUFFIX,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        r = requests.get(_PUBMED_ESEARCH, params=search_params, headers=_PUBMED_HEADERS, timeout=15)
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        logger.warning(f"PubMed esearch failed: {e}")
        return []

    if not pmids:
        return []

    # Step 2: efetch – get abstracts in plain text
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "text",
    }
    try:
        time.sleep(0.34)   # stay under 3 req/s limit
        r = requests.get(_PUBMED_EFETCH, params=fetch_params, headers=_PUBMED_HEADERS, timeout=20)
        raw_text = r.text.strip()
    except Exception as e:
        logger.warning(f"PubMed efetch failed: {e}")
        return []

    # Step 3: split into per-abstract then per-chunk pieces
    results = []
    # NCBI plain-text separates records with blank lines before PMID:
    abstracts = re.split(r'\n{2,}(?=\d+\.)', raw_text)
    for i, abstract in enumerate(abstracts):
        abstract = abstract.strip()
        if not abstract:
            continue
        pmid = pmids[i] if i < len(pmids) else "unknown"
        for chunk in _split_into_chunks(abstract, chunk_chars):
            results.append({
                "text": chunk,
                "source": f"PubMed:PMID{pmid}",
            })
    return results


# ──────────────────────────────────────────
# SEMANTIC SCHOLAR
# ──────────────────────────────────────────

_S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_HEADERS = {
    "User-Agent": "ADRD-RAG-Research/1.0 (academic research; contact: adrd-rag@research.local)"
}


def search_semantic_scholar(query: str, max_results: int = 5, chunk_chars: int = 1200) -> list[dict]:
    """
    Search Semantic Scholar for paper abstracts (completely free, no key).
    Rate limit: 100 req/5 min unauthenticated.
    Returns: [{"text": str, "source": str}, ...]
    """
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,externalIds",
    }
    try:
        time.sleep(0.1)
        r = requests.get(_S2_SEARCH, params=params, headers=_S2_HEADERS, timeout=15)
        papers = r.json().get("data", [])
    except Exception as e:
        logger.warning(f"Semantic Scholar search failed: {e}")
        return []

    results = []
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        title  = paper.get("title", "unknown")
        paper_id = paper.get("paperId", "unknown")
        source = f"S2:{paper_id}|{title[:60]}"
        for chunk in _split_into_chunks(abstract, chunk_chars):
            results.append({
                "text": chunk,
                "source": source,
            })
    return results


# ──────────────────────────────────────────
# COMBINED WEB AUGMENTATION
# ──────────────────────────────────────────

def web_augment(
    query: str,
    nli_model,  # Left for backwards compatibility, but logic bypassed
    cross_encoder,
    relevance_threshold: float = 0.3,
    max_web_passages: int = 10,
    max_wiki_pages: int = 3,
    max_pubmed_results: int = 5,
    max_s2_results: int = 5,
    already_seen_ids: set = None,
    domain: str = "general",  # "caregiving" | "medical" | "general"
) -> tuple[list[str], list[str], list[float]]:
    """
    Fetch supplementary passages from domain-appropriate sources.

    Domain routing (No Tavily / Paid APIs required):
      "caregiving" -> alz.org + nia.nih.gov + Wikipedia (caregiver focus)
      "medical"    -> PubMed + Semantic Scholar + Wikipedia (clinical focus)
      "general"    -> Wikipedia + PubMed + Semantic Scholar (standard fallback)
    """

    if already_seen_ids is None:
        already_seen_ids = set()

    # 1. Domain-aware source gathering
    candidates = []
    if domain == "caregiving":
        # Caregiving questions: Prioritize authoritative caregiver guides via DuckDuckGo site filter
        logger.info(f"web_augment domain=caregiving: DDG(site:alz.org/nia.nih.gov/caregiver.org) + Wikipedia")
        ddg_results = search_duckduckgo_domain(query, domain="caregiving", max_results=max_pubmed_results)
        if ddg_results:
            candidates.extend(ddg_results)
        else:
            # Fallback to direct scrapers if DDG not available or returns nothing
            candidates.extend(search_alz_org(query, max_results=max_pubmed_results))
            candidates.extend(search_nia_nih(query, max_results=max_pubmed_results))
        candidates.extend(search_wikipedia(query, max_pages=max_wiki_pages))
        
    elif domain == "medical":
        # Medical facts / TF questions: Prioritize peer-reviewed literature via DDG site filter
        logger.info(f"web_augment domain=medical: DDG(site:merckmanuals/mayoclinic/ncbi) + PubMed + S2 + Wikipedia")
        ddg_results = search_duckduckgo_domain(query, domain="medical", max_results=max_s2_results)
        if ddg_results:
            candidates.extend(ddg_results)
        else:
            candidates.extend(search_pubmed(query, max_results=max_pubmed_results))
            candidates.extend(search_semantic_scholar(query, max_results=max_s2_results))
        candidates.extend(search_wikipedia(query, max_pages=max_wiki_pages))
        
    else:
        # General (default): Wikipedia + PubMed + S2
        logger.info(f"web_augment domain=general: Wikipedia + PubMed + S2")
        candidates.extend(search_wikipedia(query, max_pages=max_wiki_pages))
        candidates.extend(search_pubmed(query, max_results=max_pubmed_results))
        candidates.extend(search_semantic_scholar(query, max_results=max_s2_results))

    logger.info(f"web_augment raw candidates: {len(candidates)} (domain={domain})")

    if not candidates:
        return [], [], []

    # 2. Cross-encoder relevance scoring
    logit_to_prob = lambda x: 1 / (1 + math.exp(-max(min(float(x), 100), -100)))
    texts  = [c["text"] for c in candidates]
    logits = cross_encoder.predict([[query, t[:1000]] for t in texts])
    scored = [
        (candidates[i], logit_to_prob(logits[i]))
        for i in range(len(candidates))
    ]
    
    # Filter by threshold and sort best-first
    scored = [(c, s) for c, s in scored if s >= relevance_threshold]
    scored.sort(key=lambda x: x[1], reverse=True)

    # 3. Compile final passages (NLI check completely removed to prevent false rejections)
    passages, sources, scores = [], [], []
    for candidate, rel_score in scored:
        if len(passages) >= max_web_passages:
            break

        text   = candidate["text"]
        source = candidate["source"]
        pid    = f"{source}__{hash(text)}"
        
        if pid in already_seen_ids:
            continue

        passages.append(text)
        sources.append(source)
        scores.append(rel_score)
        already_seen_ids.add(pid)

    logger.info(
        f"web_augment final: {len(passages)} passages kept (threshold={relevance_threshold}, cap={max_web_passages})"
    )
    return passages, sources, scores