# web_fallback_retriever.py
from __future__ import annotations
import os, re, json, time, hashlib
from urllib.parse import urlparse
from typing import List, Tuple, Dict, Optional

import requests

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _allowed(url: str, allow_domains: List[str]) -> bool:
    d = _domain(url)
    if not d:
        return False
    for ad in allow_domains:
        ad = ad.lower()
        if d == ad or d.endswith("." + ad):
            return True
    return False

def _clean_whitespace(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


class WebFallbackRetriever:
    """
    FREE research-only web fallback:
      1) PubMed E-utilities (official, free)
      2) Wikipedia API (free)
      3) DuckDuckGo search (unofficial, free; optional) -> fetch allowed domains only
    """
    def __init__(self, allow_domains: List[str], cache_dir: str, timeout_sec: int = 20, sleep_sec: float = 0.2):
        self.allow_domains = allow_domains
        self.cache_dir = cache_dir
        self.timeout_sec = timeout_sec
        self.sleep_sec = sleep_sec
        os.makedirs(self.cache_dir, exist_ok=True)

    # -------- cache --------
    def _cache_path(self, key: str) -> str:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def _cache_get(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _cache_set(self, key: str, obj: dict):
        p = self._cache_path(key)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # -------- PubMed --------
    def pubmed_search(self, query: str, retmax: int = 5) -> List[str]:
        key = f"pubmed_search::{retmax}::{query}"
        cached = self._cache_get(key)
        if cached and "pmids" in cached:
            return cached["pmids"]

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": str(retmax)}
        r = requests.get(url, params=params, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        pmids = data.get("esearchresult", {}).get("idlist", []) or []
        self._cache_set(key, {"pmids": pmids})
        return pmids

    def pubmed_fetch_summaries(self, pmids: List[str]) -> List[Tuple[str, str]]:
        if not pmids:
            return []
        key = f"pubmed_sum::{','.join(pmids)}"
        cached = self._cache_get(key)
        if cached and "items" in cached:
            return [(x["url"], x["text"]) for x in cached["items"]]

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
        r = requests.get(url, params=params, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})

        out = []
        for pid in pmids:
            item = result.get(pid, {})
            title = item.get("title", "")
            journal = item.get("fulljournalname", "")
            pubdate = item.get("pubdate", "")
            url_ = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            if not _allowed(url_, self.allow_domains):
                continue
            text = _clean_whitespace(f"PubMed record. Title: {title}. Journal: {journal}. Date: {pubdate}.")
            if text:
                out.append((url_, text))

        self._cache_set(key, {"items": [{"url": u, "text": t} for u, t in out]})
        return out

    # -------- Wikipedia --------
    def wikipedia_search_titles(self, query: str, limit: int = 3) -> List[str]:
        key = f"wiki_search::{limit}::{query}"
        cached = self._cache_get(key)
        if cached and "titles" in cached:
            return cached["titles"]

        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": str(limit)}
        r = requests.get(url, params=params, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        titles = [x["title"] for x in data.get("query", {}).get("search", [])]
        self._cache_set(key, {"titles": titles})
        return titles

    def wikipedia_fetch_extracts(self, titles: List[str]) -> List[Tuple[str, str]]:
        out = []
        url = "https://en.wikipedia.org/w/api.php"
        for t in titles:
            key = f"wiki_extract::{t}"
            cached = self._cache_get(key)
            if cached and "text" in cached and "url" in cached:
                out.append((cached["url"], cached["text"]))
                continue

            params = {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "titles": t, "format": "json"}
            r = requests.get(url, params=params, timeout=self.timeout_sec)
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for _, p in pages.items():
                title = p.get("title", "")
                extract = _clean_whitespace(p.get("extract", ""))
                if extract:
                    src = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    if _allowed(src, self.allow_domains):
                        out.append((src, extract))
                        self._cache_set(key, {"url": src, "text": extract})
        return out

    # -------- DuckDuckGo (unofficial) --------
    def ddg_search_urls(self, query: str, max_results: int = 5) -> List[str]:
        if DDGS is None:
            return []

        key = f"ddg::{max_results}::{query}"
        cached = self._cache_get(key)
        if cached and "urls" in cached:
            return cached["urls"]

        urls = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    u = r.get("href") or r.get("url")
                    if u:
                        urls.append(u)
        except Exception:
            urls = []

        self._cache_set(key, {"urls": urls})
        return urls

    def fetch_page_text(self, url: str, max_chars: int = 12000) -> str:
        if not _allowed(url, self.allow_domains):
            return ""
        key = f"fetch::{url}"
        cached = self._cache_get(key)
        if cached and "text" in cached:
            return cached["text"]

        try:
            if trafilatura:
                downloaded = trafilatura.fetch_url(url, timeout=self.timeout_sec)
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) if downloaded else ""
            else:
                r = requests.get(url, timeout=self.timeout_sec, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                html = r.text
                html = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
                html = re.sub(r"<style.*?</style>", " ", html, flags=re.S | re.I)
                text = re.sub(r"<[^>]+>", " ", html)

            text = _clean_whitespace(text)[:max_chars]
            self._cache_set(key, {"text": text})
            return text
        except Exception:
            return ""

    # -------- Unified retrieve --------
    def retrieve(self, queries: List[str], per_query_k: int,
                 max_page_chars: int, chunk_chars: int, chunk_overlap: int) -> Tuple[List[str], List[str]]:
        passages: List[str] = []
        sources: List[str] = []
        seen_src = set()

        for q in queries:
            q = (q or "").strip()
            if not q:
                continue

            # 1) PubMed
            try:
                pmids = self.pubmed_search(q, retmax=max(2, per_query_k))
                for src, txt in self.pubmed_fetch_summaries(pmids[:per_query_k]):
                    if src in seen_src:
                        continue
                    seen_src.add(src)
                    for c in _chunk_text(txt, chunk_chars, chunk_overlap):
                        passages.append(c)
                        sources.append(src)
            except Exception:
                pass

            # 2) Wikipedia
            try:
                titles = self.wikipedia_search_titles(q, limit=min(3, per_query_k))
                wk = self.wikipedia_fetch_extracts(titles)[:per_query_k]
                for src, txt in wk:
                    if src in seen_src:
                        continue
                    seen_src.add(src)
                    txt = (txt or "")[:max_page_chars]
                    for c in _chunk_text(txt, chunk_chars, chunk_overlap):
                        passages.append(c)
                        sources.append(src)
            except Exception:
                pass

            # 3) DuckDuckGo -> fetch allowed domains only
            try:
                for u in self.ddg_search_urls(q, max_results=per_query_k):
                    if u in seen_src:
                        continue
                    if not _allowed(u, self.allow_domains):
                        continue
                    txt = self.fetch_page_text(u, max_chars=max_page_chars)
                    if not txt:
                        continue
                    seen_src.add(u)
                    for c in _chunk_text(txt, chunk_chars, chunk_overlap):
                        passages.append(c)
                        sources.append(u)
            except Exception:
                pass

            time.sleep(self.sleep_sec)

        return passages, sources
