#!/usr/bin/env python3
"""Heterogeneous Verification Court (agent ④, upgraded from homogeneous ItV voting).

Instead of 5 identical Identify agents that can reach a confident WRONG consensus, three
SPECIALIZED expert judges cross-examine the (statement, evidence) pair, and a veto Arbiter
combines them. Each judge targets one systematic churn mode we diagnosed:
  - Entity Judge  -> ontology-boundary leakage  (TF_046: 'Alzheimer's pathology' vs generic 'dementia')
  - Modal  Judge  -> modal/degree downgrade      (TF_072: 'make them / must' vs 'helps / may')
  - Fact   Judge  -> does the evidence specifically establish the claim
Arbiter: VETO — if entity=MISMATCH or modal=DOWNGRADE, the evidence is INSUFFICIENT no matter
what the Fact judge thinks.
"""
import re
from openai import OpenAI
_c=None
def _cli():
    global _c
    if _c is None: _c=OpenAI()
    return _c
def _ask(sysp,user,mx=90,temp=0):
    r=_cli().chat.completions.create(model="gpt-4",temperature=temp,max_tokens=mx,
        messages=[{"role":"system","content":sysp},{"role":"user","content":user}])
    return (r.choices[0].message.content or "").strip()

def entity_judge(statement, evidence):
    s=("You are the ENTITY-BOUNDARY judge. The STATEMENT makes a claim about a SPECIFIC subject S1. "
       "The EVIDENCE is about a subject S2. Decide whether S2 is the SAME specific entity as S1, or a "
       "BROADER/DIFFERENT one whose general properties DO NOT transfer to S1.\n"
       "Example MISMATCH: S1='tremor due to Alzheimer's disease pathology', S2='tremor in dementia in general "
       "(often Parkinson's/Lewy-body)' -> a property of the broad class 'dementia' cannot be attributed to the "
       "specific subtype 'Alzheimer's pathology'.\n"
       "Example ALIGNED: S1='medication use in delirium', S2='medication/treatment for delirium'.\n"
       "Answer EXACTLY 'ALIGNED' or 'MISMATCH' on the first line, then one short reason.")
    t=_ask(s,f"STATEMENT: {statement}\n\nEVIDENCE: {evidence[:600]}\n\nAnswer:")
    return ("MISMATCH" if "MISMATCH" in t.upper().split("\n")[0] else "ALIGNED"), t

def modal_judge(statement, evidence):
    s=("You are the MODAL/DEGREE judge. Extract the strongest deontic/modal force in the STATEMENT "
       "(must / always / never / required / 'make them' / 'it is important to' / is-defined-as) and in the "
       "EVIDENCE (may / can / helps / beneficial / recommended / often). Force ladder (strong->weak): "
       "must/always/never/required/'make them'/'it is important to' > should > can/may/helps/beneficial.\n"
       "If the STATEMENT is an OBLIGATION/ABSOLUTE but the EVIDENCE only states a BENEFIT/POSSIBILITY -> DOWNGRADE.\n"
       "Examples:\n"
       "S='it is important to MAKE them play by the rules' | E='rules HELP engagement' -> DOWNGRADE\n"
       "S='less medication is better' | E='minimizing medication is recommended' -> MATCH\n"
       "S='delirium is very common, up to 60%' | E='30-40% develop delirium' -> MATCH (both prevalence claims)\n"
       "Answer EXACTLY one word: MATCH or DOWNGRADE.")
    t=_ask(s,f"STATEMENT: {statement}\nEVIDENCE: {evidence[:600]}\nAnswer:",4)
    return ("DOWNGRADE" if "DOWNGRA" in t.upper() else "MATCH"), t

_FACT_SYS=("You are the FACT judge. Decide if the EVIDENCE supports the statement, based on the OVERALL WEIGHT / "
    "GENERAL PRINCIPLE the literature establishes. A single nuanced or null-result study does NOT override a "
    "clearly-stated general principle (e.g. a title 'Less pharmacotherapy is more in delirium' establishes the "
    "principle even if one specific RCT showed no change). Academic abstracts that IMPLY the answer count. "
    "MEDICAL ONTOLOGY RULE: strictly distinguish Syndrome vs Disease vs Disorder, and broad class vs specific "
    "subtype. If the statement and the evidence do not correspond at the SAME ontological level, answer NO_INFO. "
    "If the evidence does not address the specific claim at all, answer NO_INFO. "
    "Reason in one sentence, then end EXACTLY with 'VERDICT: TRUE' or 'VERDICT: FALSE' or 'VERDICT: NO_INFO'.")

def _fact_once(statement, evidence, temp):
    t=_ask(_FACT_SYS,f"EVIDENCE: {evidence[:1400]}\n\nSTATEMENT: {statement}\n\nReason:",150,temp)
    m=re.search(r'VERDICT:\s*(TRUE|FALSE|NO_INFO)',t.upper())
    return m.group(1) if m else "NO_INFO"

def fact_judge(statement, evidence, n=3):
    """3-vote self-consistency majority (Fact is the high-variance judge; Entity/Modal stay single-call)."""
    from collections import Counter
    votes=[_fact_once(statement, evidence, 0.4) for _ in range(n)]
    return Counter(votes).most_common(1)[0][0], votes

def court(statement, evidence):
    """Returns (decision, flags). decision in {TRUE, FALSE, INSUFFICIENT}."""
    e,er=entity_judge(statement,evidence)
    m,mr=modal_judge(statement,evidence)
    f,fr=fact_judge(statement,evidence)
    flags={"entity":e,"modal":m,"fact":f}
    # Arbiter — VETO: any red flag overrides the fact judge
    if e=="MISMATCH" or m=="DOWNGRADE" or f=="NO_INFO":
        return "INSUFFICIENT", flags
    return f, flags   # TRUE / FALSE
