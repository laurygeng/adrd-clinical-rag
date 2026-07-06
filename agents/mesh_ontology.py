#!/usr/bin/env python3
"""Deterministic ontology-boundary gate grounded in MeSH (free, no key).

Replaces the LLM 'Entity Judge' heuristic with a reproducible check on the MeSH concept
hierarchy. A MeSH tree number (e.g. Alzheimer Disease = C10.228.140.380.100) encodes an
is-a path; a strict tree-number prefix is a strict is-a (ancestor) relation.

ontology_gate(claim_term, evidence_term) -> LEAK when the evidence concept is STRICTLY
BROADER than the claim concept (a super-class property can't transfer to a sub-type), or
when the two concepts are incomparable (siblings / unrelated branches). OK only when the
evidence is the same concept or MORE specific than the claim.
"""
import re, requests
UA = {"User-Agent": "adrd-rag/1.0"}
SPARQL = "https://id.nlm.nih.gov/mesh/sparql"
LOOKUP = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
TERM = "https://id.nlm.nih.gov/mesh/lookup/term"
_desc_cache, _tree_cache = {}, {}

def _norm(t):
    return re.sub(r"\s+", " ", re.sub(r"['’]s\b", "", (t or "").strip())).strip()

def descriptor(term):
    """Map a free-text medical term to a MeSH descriptor (id, canonical_label).
    Tries exact/contains descriptor-label match on the normalized term, then the entry-term
    (synonym) endpoint, so surface variants ("Alzheimer's disease") link to the descriptor."""
    if term in _desc_cache: return _desc_cache[term]
    q = _norm(term); res = None
    for match in ("exact", "contains"):
        try:
            r = requests.get(LOOKUP, params={"label": q, "match": match, "limit": 10},
                             headers=UA, timeout=20).json()
        except Exception:
            r = []
        if r:
            best = min(r, key=lambda x: len(x["label"]))          # prefer the most generic surface form
            res = (best["resource"].split("/")[-1], best["label"])
            break
    if res is None:                                               # fall back to entry-term (synonym) search
        try:
            r = requests.get(TERM, params={"label": q, "match": "exact", "limit": 5}, headers=UA, timeout=20).json()
            if r:
                did = (r[0].get("descriptor", "") or "").split("/")[-1] or None
                if did: res = (did, r[0].get("label", q))
        except Exception:
            pass
    _desc_cache[term] = res
    return res

def tree_numbers(did):
    """All MeSH tree numbers for a descriptor id (deterministic hierarchy positions)."""
    if did in _tree_cache: return _tree_cache[did]
    q = ("PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#> "
         "PREFIX meshv:<http://id.nlm.nih.gov/mesh/vocab#> PREFIX mesh:<http://id.nlm.nih.gov/mesh/> "
         f"SELECT DISTINCT ?tn WHERE {{ mesh:{did} meshv:treeNumber ?t . ?t rdfs:label ?tn . }}")
    try:
        r = requests.get(SPARQL, params={"query": q, "format": "JSON"}, headers=UA, timeout=30).json()
        tns = sorted(set(x["tn"]["value"] for x in r.get("results", {}).get("bindings", [])))
    except Exception:
        tns = []
    _tree_cache[did] = tns
    return tns

def relation(te, tq):
    """Relation of evidence tree-set te to claim tree-set tq."""
    if set(te) & set(tq): return "same"
    if any(q.startswith(e + ".") for e in te for q in tq): return "evidence_broader"   # te is ancestor of tq
    if any(e.startswith(q + ".") for e in te for q in tq): return "evidence_narrower"   # te is descendant of tq
    return "incomparable"

def ontology_gate(claim_term, evidence_term):
    cq, ce = descriptor(claim_term), descriptor(evidence_term)
    if not cq or not ce:
        return {"decision": "UNKNOWN", "reason": "unmapped", "claim": cq, "evidence": ce}
    tq, te = tree_numbers(cq[0]), tree_numbers(ce[0])
    if not tq or not te:
        return {"decision": "UNKNOWN", "reason": "no_tree", "claim": cq, "evidence": ce}
    rel = relation(te, tq)
    # High-precision deterministic rule: LEAK only when the evidence concept is a STRICT hypernym
    # (super-class) of the claim concept — a super-class property cannot transfer to the sub-type.
    # 'incomparable' is left to the other judges (it is too often an entity-extraction artifact on
    # relational/multi-entity claims to block deterministically).
    dec = "LEAK" if rel == "evidence_broader" else "OK"
    return {"decision": dec, "relation": rel, "claim": cq, "evidence": ce}

def extract_subject(client, text, model="gpt-4o-mini"):
    """Pull the medical subject the statement makes its claim ABOUT, at the granularity that matters
    for the ontology check: prefer the SPECIFIC subtype over a co-occurring general category."""
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=15, messages=[{"role": "user",
        "content": ("Identify the MOST SPECIFIC medical entity/subtype that this statement makes its claim ABOUT "
                    "(if both a general category and a specific subtype appear, choose the SPECIFIC subtype; if only "
                    "a general category appears, output that). Output ONLY the canonical term.\n\n"
                    f"Statement: {text}\nTerm:")}])
    return (r.choices[0].message.content or "").strip()

def entity_gate(client, statement, evidence):
    """Court-facing wrapper. Returns (verdict, detail):
      MISMATCH -> deterministic ontology-boundary leakage (evidence is a strict super-class); VETO.
      ALIGNED  -> evidence same/more-specific concept; passes.
      DEFER    -> concept(s) unmapped in MeSH; fall back to the LLM entity judge (no deterministic call)."""
    cs = extract_subject(client, statement)
    es = extract_subject(client, evidence)
    g = ontology_gate(cs, es)
    if g["decision"] == "LEAK":
        return "MISMATCH", {"claim_subj": cs, "evidence_subj": es, **g}
    if g["decision"] == "OK":
        return "ALIGNED", {"claim_subj": cs, "evidence_subj": es, **g}
    return "DEFER", {"claim_subj": cs, "evidence_subj": es, **g}
