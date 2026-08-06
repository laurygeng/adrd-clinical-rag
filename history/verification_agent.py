#!/usr/bin/env python3
"""Heterogeneous Verification agent

Three SPECIALIZED expert judges cross-examine the (question/claim, evidence) pair.
Arbiter: VETO — if entity=MISMATCH or modal=DOWNGRADE, the evidence is INSUFFICIENT.
"""
import re
from openai import OpenAI

_c = None
def _cli():
    global _c
    if _c is None: _c = OpenAI()
    return _c

def _ask(sysp, user, mx=90, temp=0):
    r = _cli().chat.completions.create(
        model="gpt-4o", # Upgraded to gpt-4o for speed and lower cost
        temperature=temp,
        max_tokens=mx,
        messages=[{"role": "system", "content": sysp}, {"role": "user", "content": user}]
    )
    return (r.choices[0].message.content or "").strip()

def entity_judge(statement, evidence):
    s = ("You are the ENTITY-BOUNDARY judge. The TARGET (Question/Claim) is about a SPECIFIC subject S1. "
         "The EVIDENCE is about a subject S2. Decide whether S2 is the SAME specific entity as S1, or a "
         "BROADER/DIFFERENT one whose general properties DO NOT transfer to S1.\n"
         "Example MISMATCH: S1='Alzheimer's pathology', S2='dementia in general'.\n"
         "Answer EXACTLY 'ALIGNED' or 'MISMATCH' on the first line, then one short reason.")
    t = _ask(s, f"TARGET: {statement}\n\nEVIDENCE: {evidence[:600]}\n\nAnswer:")
    return ("MISMATCH" if "MISMATCH" in t.upper().split("\n")[0] else "ALIGNED"), t

def modal_judge(statement, evidence):
    s = ("You are the MODAL/DEGREE judge. Extract the strongest deontic/modal force in the TARGET "
         "and the EVIDENCE. Force ladder: must/always/required > should > can/may/helps.\n"
         "If the TARGET asks about an OBLIGATION but the EVIDENCE only states a POSSIBILITY -> DOWNGRADE.\n"
         "If they align or the target is just a general question -> MATCH.\n"
         "Answer EXACTLY one word: MATCH or DOWNGRADE.")
    t = _ask(s, f"TARGET: {statement}\nEVIDENCE: {evidence[:600]}\nAnswer:", 4)
    return ("DOWNGRADE" if "DOWNGRA" in t.upper() else "MATCH"), t

_FACT_SYS = (
    "You are the FACT judge. Decide if the EVIDENCE specifically addresses and supports the TARGET "
    "(Question or Claim). "
    "If the evidence does not address the specific target at all, answer NO_INFO. "
    "Reason in one sentence, then end EXACTLY with 'VERDICT: TRUE', 'VERDICT: FALSE', or 'VERDICT: NO_INFO'."
)

def _fact_once(statement, evidence, temp):
    t = _ask(_FACT_SYS, f"EVIDENCE: {evidence[:1400]}\n\nTARGET: {statement}\n\nReason:", 150, temp)
    m = re.search(r'VERDICT:\s*(TRUE|FALSE|NO_INFO)', t.upper())
    return m.group(1) if m else "NO_INFO"

def fact_judge(statement, evidence, n=3):
    from collections import Counter
    votes = [_fact_once(statement, evidence, 0.4) for _ in range(n)]
    return Counter(votes).most_common(1)[0][0], votes

def _entity_check(statement, evidence, use_mesh=True):
    if use_mesh:
        try:
            from core.mesh_ontology import entity_gate
            v, det = entity_gate(_cli(), statement, evidence)
            if v in ("MISMATCH", "ALIGNED"):
                return v, {"source": "mesh", **det}
        except Exception:
            pass
    e, er = entity_judge(statement, evidence)
    return e, {"source": "llm", "text": er}

def court(statement, evidence, use_mesh=True):
    """Returns (decision, flags). decision in {TRUE, FALSE, INSUFFICIENT}."""
    e, edet = _entity_check(statement, evidence, use_mesh)
    m, mr = modal_judge(statement, evidence)
    f, fr = fact_judge(statement, evidence)
    flags = {"entity": e, "modal": m, "fact": f, "entity_src": edet.get("source")}
    
    if e == "MISMATCH" or m == "DOWNGRADE" or f == "NO_INFO":
        return "INSUFFICIENT", flags
    return f, flags