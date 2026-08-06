#!/usr/bin/env python3
"""
Step 1 — validate Identify-then-Verify (ItV, EACL-2026 'Knowing What's Missing')
as a SUFFICIENCY gate for MC, where NLI/prompt failed (NLI AUC ~0.45).

ItV per question:
  Step 1 Identify (N=5, temp>0): ask what single piece of info is still MISSING to
         determine the correct answer (or 'NONE' if context is sufficient).
  Step 2a Consensus (local embeddings, free): pick the gap statement most central
         (highest avg cosine sim) among the non-NONE hypotheses.
  Step 2b Verify (1 call, temp=0): is that consensus gap PRESENT or ABSENT in context?
         present => the gap was hallucinated => SUFFICIENT; absent => INSUFFICIENT.

Signal for the gate: a continuous sufficiency score; we measure AUC(score -> Is_Correct)
and compare to NLI (0.45) and the contingency of ItV's binary decision vs correctness.
Uses gpt-4o-mini (cheap) for identify/verify; all-MiniLM for consensus (local).
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai import OpenAI
from sentence_transformers import SentenceTransformer

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RET = os.path.join(CODE, "retrieval_results", "retrieval_ADRD_all_MERGED_corrected_bucketA.csv")
GEN = os.path.join(CODE, "generate", "answers", "answers_gpt4_ADRD_all_rag_20260617_205522.csv")
MC_JSON = os.path.join(CODE, "data", "ADRD_Caregiving_Multiple_Choice.json")
MODEL = "gpt-4o-mini"
N = 5

client = OpenAI()
emb = SentenceTransformer("all-MiniLM-L6-v2")

def auc(y, s):
    y = np.array(y).astype(int); s = np.array(s, float)
    P, Ng = s[y == 1], s[y == 0]
    return float((P[:, None] > Ng[None, :]).mean() + 0.5 * (P[:, None] == Ng[None, :]).mean()) if len(P) and len(Ng) else float("nan")

def identify(qblock, context, n=N):
    sysp = ("You judge whether a CONTEXT is sufficient to answer a multiple-choice question. "
            "Name the SINGLE most important piece of information still MISSING from the context that is "
            "needed to confidently determine the correct option. If the context already contains everything "
            "needed, answer exactly 'NONE'. Answer with one short phrase only.")
    outs = []
    for _ in range(n):
        r = client.chat.completions.create(model=MODEL, temperature=0.7, max_tokens=40,
            messages=[{"role": "system", "content": sysp},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{qblock}"}])
        outs.append(r.choices[0].message.content.strip())
    return outs

def verify(claim, context):
    sysp = "Decide if a specific piece of information is PRESENT in the context. Answer exactly 'PRESENT' or 'ABSENT'."
    r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=4,
        messages=[{"role": "system", "content": sysp},
                  {"role": "user", "content": f"Context:\n{context}\n\nInformation: {claim}\n\nIs this present in the context?"}])
    return "PRESENT" if "PRESENT" in r.choices[0].message.content.strip().upper() else "ABSENT"

def is_none(s): return s.strip().upper().startswith("NONE") or s.strip() == ""

def main():
    ret = pd.read_csv(RET).set_index("Question_ID")
    gen = pd.read_csv(GEN).set_index("Question_ID")
    ret = ret.join(gen["Is_Correct"])
    mc_items = {f"ADRD_MC_{it['ID']:03d}": it for it in json.load(open(MC_JSON))["data"]}

    rows = []
    for qid, r in ret[ret.Type == "MC"].iterrows():
        it = mc_items[qid]
        qblock = it["Question"] + "\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in it["Options"].items())
        ctx = "\n\n".join(json.loads(r["Retrieved_Passages"])[:8])
        gaps = identify(qblock, ctx)
        none_frac = float(np.mean([is_none(g) for g in gaps]))
        real = [g for g in gaps if not is_none(g)]
        if none_frac >= 0.6 or not real:
            suff, vstate = 1, "consensus-sufficient"
        else:
            E = emb.encode(real, normalize_embeddings=True)
            avg = (E @ E.T).mean(1)
            consensus = real[int(avg.argmax())]
            v = verify(consensus, ctx)
            suff, vstate = (1 if v == "PRESENT" else 0), v
        # continuous sufficiency score in [0,1]: self-consistency, adjusted by verification
        score = none_frac if suff == 1 else min(none_frac, 0.3)
        rows.append({"qid": qid, "is_correct": int(bool(r["Is_Correct"])),
                     "none_frac": none_frac, "itv_suff": suff, "itv_score": score, "vstate": vstate})
        print(f"  {qid}: none_frac={none_frac:.1f} suff={suff} ({vstate}) correct={int(bool(r['Is_Correct']))}", flush=True)

    d = pd.DataFrame(rows)
    print("\n===== ItV on MC (n=%d) =====" % len(d))
    print(f"  ItV sufficiency score -> correct : AUC = {auc(d.is_correct, d.itv_score):.3f}   (NLI was 0.45, GPT-4o-conf n/a)")
    print(f"  (none_frac alone)     -> correct : AUC = {auc(d.is_correct, d.none_frac):.3f}")
    print("\n  ItV binary sufficiency vs answer-correctness:")
    print(pd.crosstab(d.itv_suff, d.is_correct, rownames=['ItV_sufficient'], colnames=['Answer_correct']))
    print(f"\n  When ItV says INSUFFICIENT, GPT-4o was wrong: {d[(d.itv_suff==0)].is_correct.eq(0).sum()}/{(d.itv_suff==0).sum()}")
    print(f"  When ItV says SUFFICIENT,   GPT-4o was right: {d[(d.itv_suff==1)].is_correct.eq(1).sum()}/{(d.itv_suff==1).sum()}")

if __name__ == "__main__":
    main()
