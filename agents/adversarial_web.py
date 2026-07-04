#!/usr/bin/env python3
"""Adversarial Web-Research Agent (agent ⑥, upgraded).

A single confirmation-biased query re-runs the TF_035 failure (search 'recovery' -> get recovery
content). This agent has a DUAL PERSONALITY: it fires BOTH a support-seeking and a refutation-
seeking query, and writes web_pro / web_con back to the blackboard so the Court must judge amid
conflicting evidence instead of a one-sided echo.

Backend: Tavily when TAVILY_API_KEY is set (clean content), else DuckDuckGo + PubMed via the
existing WebFallbackRetriever. Returns (pro_snippets, con_snippets).
"""
import os, sys, requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from openai import OpenAI

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

def _tavily(q, key, n=4):
    try:
        d = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": q,
            "search_depth": "advanced", "max_results": n, "include_answer": True}, timeout=40).json()
        out = [(d.get("answer") or "")] + [x.get("content", "") for x in d.get("results", [])]
        return [s for s in out if s.strip()]
    except Exception:
        return []

def _ddg(retriever, q, n=4):
    try:
        wp, ws = retriever.retrieve(queries=[q], per_query_k=n, max_sentences_per_source=8)
        return wp[:12]
    except Exception:
        return []

def research(client, statement, retriever=None):
    """Returns (pro_snippets, con_snippets, (pro_q, con_q))."""
    pro_q, con_q = _dual_queries(client, statement)
    key = os.environ.get("TAVILY_API_KEY")
    if key:
        return _tavily(pro_q, key), _tavily(con_q, key), (pro_q, con_q)
    return _ddg(retriever, pro_q), _ddg(retriever, con_q), (pro_q, con_q)
