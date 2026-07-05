#!/usr/bin/env python3
"""Adversarial Web-Research Agent (agent ⑥, upgraded).

A single confirmation-biased query re-runs the TF_035 failure (search 'recovery' -> get recovery
content). This agent has a DUAL PERSONALITY: it fires BOTH a support-seeking and a refutation-
seeking query and writes web_pro / web_con to the blackboard, so the Court judges amid conflicting
evidence instead of a one-sided echo.

Backends (pluggable, all free):
  Track B — general/policy : Exa (clean full-context markdown; preserves modal context)
  Track A — medical facts  : Europe PMC (structured abstracts, no key) + Semantic Scholar
  Fallbacks                : Tavily (if key), DuckDuckGo + PubMed via WebFallbackRetriever
"""
import os, sys, requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from openai import OpenAI

UA = {"User-Agent": "adrd-medical-rag/1.0"}

def _dual_queries(client, statement, model="gpt-4o-mini"):
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=80,
        messages=[{"role": "user", "content":
            "Given a True/False statement, output TWO web search queries on two lines:\n"
            "Line 1 (SUPPORT): a query to find evidence the statement is TRUE.\n"
            "Line 2 (REFUTE): a query to find evidence it is FALSE or the medical consensus/limits against it.\n"
            f"Statement: {statement}\nQueries:"}])
    lines = [l.strip().lstrip("12.:- ").strip() for l in (r.choices[0].message.content or "").splitlines() if l.strip()]
    pro = lines[0] if lines else statement
    con = lines[1] if len(lines) > 1 else f"evidence against: {statement}"
    return pro, con

def _exa(query, n=3):
    key = os.environ.get("EXA_API_KEY")
    if not key: return []
    try:
        d = requests.post("https://api.exa.ai/search", headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"query": query, "numResults": n, "contents": {"text": {"maxCharacters": 600}}}, timeout=30).json()
        return [(x.get("text") or "").strip() for x in d.get("results", []) if (x.get("text") or "").strip()]
    except Exception:
        return []

def _europepmc(query, n=3):
    try:
        d = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": n, "resultType": "core"}, headers=UA, timeout=30).json()
        import re
        out = []
        for x in d.get("resultList", {}).get("result", []):
            t = x.get("abstractText") or x.get("title") or ""
            t = re.sub(r"<[^>]+>", " ", t).strip()      # strip <h4> etc.
            if t: out.append(t[:600])
        return out
    except Exception:
        return []

def _tavily(q, key, n=4):
    try:
        d = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": q,
            "search_depth": "advanced", "max_results": n, "include_answer": True}, timeout=40).json()
        return [s for s in [(d.get("answer") or "")] + [x.get("content", "") for x in d.get("results", [])] if s.strip()]
    except Exception:
        return []

def _gather(query, retriever):
    """Collect evidence for one query across the configured free backends."""
    ev = _exa(query, 3) + _europepmc(query, 3)                     # Exa (general) + Europe PMC (medical)
    if not ev and os.environ.get("TAVILY_API_KEY"):
        ev = _tavily(query, os.environ["TAVILY_API_KEY"])
    if not ev and retriever is not None:
        try:
            wp, _ = retriever.retrieve(queries=[query], per_query_k=4, max_sentences_per_source=8); ev = wp[:10]
        except Exception:
            ev = []
    return ev

def research(client, statement, retriever=None):
    """Returns (pro_snippets, con_snippets, (pro_q, con_q))."""
    pro_q, con_q = _dual_queries(client, statement)
    return _gather(pro_q, retriever), _gather(con_q, retriever), (pro_q, con_q)
