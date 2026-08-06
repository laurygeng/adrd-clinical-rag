#!/usr/bin/env python3
"""Surgical additive supplementation for TF (retrieval-side method, model-agnostic).

The KEY problem with the old TF web fallback was FLOODING: merging ~12 web sentences into the
context destabilized the binary judgment (churn), so web was disabled for TF entirely. This
method instead *surgically* fetches only the one verified fact needed to check the statement and
APPENDS it to the full local context — additive, not fragmenting; a single fact, not a flood.

Steps: statement -> LLM verification query -> Tavily (answer + top snippets) -> append to context.
"""
import os, requests
from openai import OpenAI

def gen_query(client, statement, model="gpt-4o-mini"):
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=40,
        messages=[{"role": "user", "content": f"Write ONE concise web search query to verify whether this statement is true.\nStatement: {statement}\nQuery:"}])
    return (r.choices[0].message.content or "").strip().strip('"')

def tavily(query, key=None, max_results=4):
    key = key or os.environ.get("TAVILY_API_KEY")
    if not key: return "", []
    try:
        d = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": query,
            "search_depth": "advanced", "max_results": max_results, "include_answer": True}, timeout=40).json()
        return (d.get("answer", "") or ""), [x.get("content", "") for x in d.get("results", [])]
    except Exception:
        return "", []

def supplement_context(client, statement, local_context, key=None):
    """Return context enriched with a surgically-fetched verified fact (or unchanged if none)."""
    ans, contents = tavily(gen_query(client, statement), key=key)
    if not ans and not contents:
        return local_context
    add = ans + "\n" + "\n".join(c[:350] for c in contents[:2])
    return local_context + "\n\n[Additional verified evidence]\n" + add
