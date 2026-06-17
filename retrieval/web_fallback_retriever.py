# web_fallback_retriever.py
from __future__ import annotations
import os, re, json, time, hashlib
from urllib.parse import urlparse
from typing import List, Tuple, Dict, Optional
import requests
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    import trafilatura
except Exception:
    trafilatura = None

# [Fix]: Support different versions of the DuckDuckGo package to prevent silent failures
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


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

def _chunk_text_by_sentences(text: str, max_chars: int = 15000) -> List[str]:
    """
    [Core Refactoring]: Utilize NLTK to tokenize web pages and abstracts into independent, 
    clean sentences, completely avoiding paragraph-level noise.
    """
    text = (text or "").strip()
    if not text:
        return []
    
    # Truncate long web texts to prevent memory issues
    truncated_text = text[:max_chars]
    
    # Precise sentence tokenization
    raw_sentences = nltk.tokenize.sent_tokenize(truncated_text)
    
    clean_sentences = []
    for s in raw_sentences:
        s_clean = _clean_whitespace(s)
        # Filter out overly short, meaningless characters or navigation bar residues
        if len(s_clean) > 15:
            clean_sentences.append(s_clean)
            
    return clean_sentences


class WebFallbackRetriever:
    """
    Enhanced research-only fallback retriever with PubMed Direct Integration,
    Anti-Bot web scraping, and Sentence-Level high-purity chunking.
    """
    def __init__(self, allow_domains=None, cache_dir: str = "", timeout_sec: int = 20, sleep_sec: float = 0.5):
        self.allow_domains = allow_domains or []
        self.cache_dir = cache_dir
        self.timeout_sec = timeout_sec
        self.sleep_sec = sleep_sec
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    # -------- Cache Management --------
    def _cache_path(self, key: str) -> str:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def _cache_get(self, key: str) -> Optional[dict]:
        if not self.cache_dir: return None
        p = self._cache_path(key)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _cache_set(self, key: str, obj: dict):
        if not self.cache_dir: return
        p = self._cache_path(key)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ==========================================
    # PubMed API
    # ==========================================
    def pubmed_fetch_abstracts(self, query: str, max_results: int = 3) -> List[Tuple[str, str]]:
        key = f"pubmed_search_v3::{max_results}::{query}" # Upgraded to v3 to forcefully bypass old corrupted cache
        cached = self._cache_get(key)
        # [Fix]: Strictly prohibit the use of empty caches
        if cached and "items" in cached and cached["items"]:
            return [(x["url"], x["text"]) for x in cached["items"]]

        results = []
        try:
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": str(max_results)}
            search_resp = requests.get(search_url, params=params, timeout=self.timeout_sec).json()
            id_list = search_resp.get("esearchresult", {}).get("idlist", [])
            
            if id_list:
                ids_str = ",".join(id_list)
                fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                fetch_params = {"db": "pubmed", "id": ids_str, "retmode": "json"}
                fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=self.timeout_sec).json()
                summaries = fetch_resp.get("result", {})
                
                for pid in id_list:
                    item = summaries.get(pid, {})
                    title = item.get("title", "")
                    source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
                    if title:
                        results.append((source_url, f"PubMed Direct Evidence: {title}"))
        except Exception:
            pass

        # [Fix]: Only save to cache if actual data is retrieved successfully
        if results:
            self._cache_set(key, {"items": [{"url": u, "text": t} for u, t in results]})
        return results

    # ==========================================
    # DuckDuckGo Web Search
    # ==========================================
    def ddg_search_urls(self, query: str, max_results: int = 5) -> List[str]:
        if not DDGS: 
            print("⚠️ WARNING: DuckDuckGo library missing. Run 'pip install duckduckgo-search'")
            return []
            
        key = f"ddg_v3::{max_results}::{query}"
        cached = self._cache_get(key)
        # [Fix]: Strictly prohibit the use of empty caches
        if cached and "urls" in cached and cached["urls"]:
            return cached["urls"]

        urls = []
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                if results:
                    for r in results:
                        u = r.get("href")
                        if u: urls.append(u)
        except Exception as e:
            pass
            
        if urls:
            self._cache_set(key, {"urls": urls})
        return urls

    # ==========================================
    # Web Scraping
    # ==========================================
    def fetch_page_text(self, url: str, max_chars: int = 25000) -> str:
        key = f"fetch_v3::{url}"
        cached = self._cache_get(key)
        # [Fix]: Strictly prohibit the use of empty caches
        if cached and "text" in cached and cached["text"]:
            return cached["text"]

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        text_clean = ""
        # 1. Try Trafilatura for clean extraction
        if trafilatura:
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                    if text:
                        text_clean = _clean_whitespace(text)[:max_chars]
            except Exception:
                pass

        # 2. Fallback to BeautifulSoup if Trafilatura fails
        if not text_clean and BeautifulSoup:
            try:
                resp = requests.get(url, headers=headers, timeout=self.timeout_sec)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.extract()
                    text = soup.get_text(separator=' ', strip=True)
                    if text:
                        text_clean = _clean_whitespace(text)[:max_chars]
            except Exception:
                pass
                
        # [Fix]: Only save to cache if actual text is extracted successfully
        if text_clean:
            self._cache_set(key, {"text": text_clean})
            
        return text_clean

    def retrieve(self, queries: List[str], per_query_k: int = 5, max_page_chars: int = 25000, chunk_chars: int = 2000, chunk_overlap: int = 200) -> Tuple[List[str], List[str]]:
        passages = []
        sources = []
        seen_src = set()

        for q in queries:
            if not q.strip(): continue
            
            # 1) PubMed Direct Summary
            try:
                pubmed_data = self.pubmed_fetch_abstracts(q, max_results=per_query_k)
                for src, txt in pubmed_data:
                    if src in seen_src: continue
                    seen_src.add(src)
                    for s_chunk in _chunk_text_by_sentences(txt, max_chars=max_page_chars):
                        passages.append(s_chunk)
                        sources.append(src)
            except Exception:
                pass

            # 2) DuckDuckGo
            try:
                for u in self.ddg_search_urls(q, max_results=per_query_k):
                    if u in seen_src: continue
                    if not _allowed(u, self.allow_domains): continue
                    
                    txt = self.fetch_page_text(u, max_chars=max_page_chars)
                    if not txt: continue
                        
                    seen_src.add(u)
                    for s_chunk in _chunk_text_by_sentences(txt, max_chars=max_page_chars):
                        passages.append(s_chunk)
                        sources.append(u)
            except Exception:
                pass

            time.sleep(self.sleep_sec)

        return passages, sources