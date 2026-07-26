#!/usr/bin/env python3
"""Per-option conditioned retrieval for MC (a retrieval-side method, model-agnostic).

Instead of one query for the whole question, retrieve evidence FOR EACH OPTION separately
(query = question stem + that option), then score how strongly the option-conditioned evidence
supports that option being the answer, and pick the best-supported option. Targets the failure
where a single query surfaces evidence for a distractor and buries the correct option's evidence.

Scoring uses the SAME frozen generator (gpt-4) as a grounded comparator — the contribution is the
per-option conditioned RETRIEVAL, not a stronger model."""
import re
from openai import OpenAI

def parse_mc(question):
    """Return (stem, {letter: option_text})."""
    stem = re.split(r'\n\s*Options?\s*:', question, flags=re.I)[0].strip()
    opts = {}
    for m in re.finditer(r'(?m)^\s*([A-E])[\.\)]\s*(.+?)\s*$', question):
        t = m.group(2).strip()
        if t:
            opts[m.group(1)] = t
    return stem, opts

def _score(client, stem, opt_text, ctx, model="gpt-4"):
    sys = ("You judge, based ONLY on the evidence, how strongly it supports that the candidate "
           "answer is the correct answer to the question. Output a SINGLE integer 0-10 "
           "(10 = evidence directly supports it; 0 = no support or contradicted). No other text.")
    user = f"Evidence:\n{ctx}\n\nQuestion: {stem}\nCandidate answer: {opt_text}\n\nSupport score (0-10):"
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=3,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}])
    m = re.search(r'\d+', r.choices[0].message.content or "")
    return int(m.group()) if m else 0

def answer_mc_per_option(retriever, build_ctx, question, top_k=6, pre_k=30, model="gpt-4", client=None):
    """Returns (chosen_letter, scores_dict). retriever = AdvancedRetriever; build_ctx = fn(passages,n)->str."""
    client = client or OpenAI()
    stem, opts = parse_mc(question)
    if len(opts) < 2:
        return None, {}
    scores = {}
    for letter, otext in opts.items():
        q = f"{stem} {otext}"
        ps, _, _ = retriever.get_retrieved_passages(q, top_k=top_k, bm25_weight=0.3, vector_weight=0.7,
                                                    pre_k=pre_k, window_size=800)
        ctx = build_ctx(ps, top_k)
        scores[letter] = _score(client, stem, otext, ctx, model=model)
    best = max(scores, key=scores.get)
    return best, scores
