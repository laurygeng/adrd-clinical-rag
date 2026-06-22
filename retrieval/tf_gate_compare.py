#!/usr/bin/env python3
"""Head-to-head on TF (120 questions, same money-run contexts):
NLI answerability gate vs Identify-then-Verify (ItV) gate.
Compares AUC(sufficiency -> Is_Correct) and false-positive rate (flagged insufficient
but the answer was actually correct => over-triggering)."""
import os, json, numpy as np, pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from openai import OpenAI

ret = pd.read_csv("../retrieval_results/retrieval_ADRD_all_MERGED_corrected_bucketA.csv").set_index("Question_ID")
gen = pd.read_csv("../generate/answers/answers_gpt4_ADRD_all_rag_20260617_205522.csv").set_index("Question_ID")
ret = ret.join(gen["Is_Correct"])
tf = ret[ret.Type == "TF"]
client = OpenAI(); MINI = "gpt-4o-mini"
nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
id2 = {int(k): v.lower() for k, v in nli.model.config.id2label.items()}; lab = {v: k for k, v in id2.items()}
ENT, CON = lab["entailment"], lab["contradiction"]
emb = SentenceTransformer("all-MiniLM-L6-v2")

def auc(y, s):
    y = np.array(y).astype(int); s = np.array(s, float); P, N = s[y == 1], s[y == 0]
    return float((P[:, None] > N[None, :]).mean() + 0.5 * (P[:, None] == N[None, :]).mean()) if len(P) and len(N) else float("nan")

def nli_suff(passages, stmt):
    pr = np.asarray(nli.predict([(p, stmt) for p in passages[:12]], apply_softmax=True, show_progress_bar=False))
    a = float(max(pr[:, ENT].max(), pr[:, CON].max()))
    return a >= 0.5, a

def itv_suff(stmt, context, n=5):
    sysp = ("You judge whether a CONTEXT is sufficient to determine if a True/False statement is correct. "
            "Name the SINGLE most important piece of information still MISSING from the context needed to "
            "confidently verify OR refute the statement. If the context already has everything needed, answer "
            "exactly 'NONE'. One short phrase only.")
    gaps = []
    for _ in range(n):
        r = client.chat.completions.create(model=MINI, temperature=0.7, max_tokens=40,
            messages=[{"role": "system", "content": sysp}, {"role": "user", "content": f"Context:\n{context}\n\nStatement: {stmt}"}])
        gaps.append((r.choices[0].message.content or "").strip())
    nf = float(np.mean([g.upper().startswith("NONE") or g == "" for g in gaps]))
    if nf >= 0.6: return True, nf
    real = [g for g in gaps if not (g.upper().startswith("NONE") or g == "")]
    if not real: return True, nf
    E = emb.encode(real, normalize_embeddings=True); cons = real[int((E @ E.T).mean(1).argmax())]
    v = client.chat.completions.create(model=MINI, temperature=0, max_tokens=4,
        messages=[{"role": "system", "content": "Is a specific piece of information PRESENT in the context? Answer 'PRESENT' or 'ABSENT'."},
                  {"role": "user", "content": f"Context:\n{context}\n\nInformation: {cons}\n\nPresent?"}])
    return ("PRESENT" in (v.choices[0].message.content or "").upper()), nf

rows = []
for qid, r in tf.iterrows():
    ps = json.loads(r["Retrieved_Passages"]); stmt = str(r["Question"]); ctx = "\n\n".join(ps[:8])
    ns, na = nli_suff(ps, stmt); isf, inf = itv_suff(stmt, ctx)
    rows.append({"correct": int(bool(r["Is_Correct"])), "nli_suff": int(ns), "nli_score": na,
                 "itv_suff": int(isf), "itv_score": inf})
d = pd.DataFrame(rows)
def fp(col):
    flagged = d[d[col] == 0]; return (flagged.correct == 1).sum(), len(flagged)
nfp, nfl = fp("nli_suff"); ifp, ifl = fp("itv_suff")
print(f"\n=== TF gate: NLI vs ItV (n={len(d)}) ===")
print(f"  AUC(score->correct):   NLI {auc(d.correct, d.nli_score):.3f}   |   ItV {auc(d.correct, d.itv_score):.3f}")
print(f"  flagged insufficient:  NLI {nfl}   |   ItV {ifl}")
print(f"  false positives (flagged but was correct):  NLI {nfp}/{nfl} = {nfp/max(nfl,1)*100:.0f}%   |   ItV {ifp}/{ifl} = {ifp/max(ifl,1)*100:.0f}%")
