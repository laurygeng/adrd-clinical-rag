#!/usr/bin/env python3
"""Dual-agent heterogeneous voting for sufficiency. Backend-pluggable (openai/gemini) so the
second agent can be swapped to Gemini later by changing one line.

Hypothesis: two DIFFERENT models (different blind spots), each 3x + semantic consensus, combined by
a cross-agent VOTE, beat one model asked 5x. Controls separate 'diversity' from 'more calls'.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai import OpenAI
from sentence_transformers import SentenceTransformer

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RET = os.path.join(CODE, "retrieval_results", "retrieval_ADRD_all_MERGED_corrected_bucketA.csv")
GEN = os.path.join(CODE, "generate", "answers", "answers_gpt4_ADRD_all_rag_20260617_205522.csv")
MC_JSON = os.path.join(CODE, "data", "ADRD_Caregiving_Multiple_Choice.json")
oai = OpenAI(); emb = SentenceTransformer("all-MiniLM-L6-v2")
_gem = None
def gem():
    global _gem
    if _gem is None:
        from google import genai; _gem = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _gem

SYS_ID = ("You judge whether a CONTEXT is sufficient to answer a multiple-choice question. Name the SINGLE most "
          "important piece of information still MISSING from the context needed to confidently determine the correct "
          "option. If the context already contains everything needed, answer exactly 'NONE'. One short phrase only.")
SYS_V = "Decide if a specific piece of information is PRESENT in the context. Answer exactly 'PRESENT' or 'ABSENT'."

def call(backend, model, sysp, user, temp, mx):
    if backend == "openai":
        r = oai.chat.completions.create(model=model, temperature=temp, max_tokens=mx,
            messages=[{"role":"system","content":sysp},{"role":"user","content":user}])
        return (r.choices[0].message.content or "").strip()
    else:  # gemini
        from google.genai import types
        import time
        for attempt in range(6):
            try:
                r = gem().models.generate_content(model=model, contents=f"{sysp}\n\n{user}",
                    config=types.GenerateContentConfig(temperature=temp, max_output_tokens=1024))
                return (r.text or "").strip()
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(6*(attempt+1)); continue
                raise
        return ""

def identify(backend, model, qb, ctx, n):
    return [call(backend, model, SYS_ID, f"Context:\n{ctx}\n\nQuestion:\n{qb}", 0.7, 40) for _ in range(n)]
def verify(backend, model, claim, ctx):
    return "PRESENT" if "PRESENT" in call(backend, model, SYS_V, f"Context:\n{ctx}\n\nInformation: {claim}\n\nPresent?", 0, 4).upper() else "ABSENT"
def is_none(s): return s.strip().upper().startswith("NONE") or not s.strip()

def verdict(gaps, ctx, backend, vmodel):
    nf = float(np.mean([is_none(g) for g in gaps])); real = [g for g in gaps if not is_none(g)]
    if nf >= 0.6 or not real: return 1, nf
    E = emb.encode(real, normalize_embeddings=True); cons = real[int((E@E.T).mean(1).argmax())]
    suff = 1 if verify(backend, vmodel, cons, ctx) == "PRESENT" else 0
    return suff, (nf if suff else min(nf, 0.3))

def auc(y, s):
    y = np.array(y).astype(int); s = np.array(s, float); P, Ng = s[y==1], s[y==0]
    return float((P[:,None]>Ng[None,:]).mean()+0.5*(P[:,None]==Ng[None,:]).mean()) if len(P) and len(Ng) else float("nan")

# ---- two agents (swap B to ("gemini","gemini-2.5-flash") later) ----
AGENT_A = ("openai", "gpt-4")
AGENT_B = ("gemini", "gemini-2.5-flash")

def main():
    ret = pd.read_csv(RET).set_index("Question_ID"); gen = pd.read_csv(GEN).set_index("Question_ID")
    ret = ret.join(gen["Is_Correct"]); mc = {f"ADRD_MC_{it['ID']:03d}": it for it in json.load(open(MC_JSON))["data"]}
    rows = []
    for qid, r in ret[ret.Type == "MC"].iterrows():
        it = mc[qid]; qb = it["Question"] + "\nOptions:\n" + "\n".join(f"{k}. {v}" for k,v in it["Options"].items())
        ctx = "\n\n".join(json.loads(r["Retrieved_Passages"])[:8])
        gA = identify(*AGENT_A, qb, ctx, 5)                       # 5 gpt-4 calls
        gB = identify(*AGENT_B, qb, ctx, 5)                       # 5 gemini calls
        s_a5 = verdict(gA, ctx, *AGENT_A)[1]
        s_b5 = verdict(gB, ctx, *AGENT_B)[1]
        sA, scA = verdict(gA[:3], ctx, *AGENT_A)
        sB, scB = verdict(gB[:3], ctx, *AGENT_B)
        rows.append({"qid": qid, "correct": int(bool(r["Is_Correct"])), "a5": s_a5, "b5": s_b5,
                     "dual_avg": (scA+scB)/2, "dual_min": min(scA,scB), "dual_vote": sA+sB+0.01*(scA+scB),
                     "sA": sA, "sB": sB})
        print(f"  {qid}: gpt4x5={s_a5:.2f} gemx5={s_b5:.2f} dual_avg={(scA+scB)/2:.2f} vote={sA+sB} correct={int(bool(r['Is_Correct']))}", flush=True)
    d = pd.DataFrame(rows)
    print(f"\n===== Sufficiency AUC on MC (n={len(d)})  A={AGENT_A[1]} B={AGENT_B[1]} =====")
    for c,name in [("a5",f"single {AGENT_A[1]} x5"),("b5",f"single {AGENT_B[1]} x5"),
                   ("dual_avg","DUAL avg"),("dual_min","DUAL min (conservative)"),("dual_vote","DUAL vote")]:
        print(f"  {name:36s}: AUC = {auc(d.correct, d[c]):.3f}")
    print(f"\n  agent agreement (sA==sB): {(d.sA==d.sB).mean():.0%}")
    
    # 替换了原有的 /tmp 路径，改为保存在项目目录下
    output_path = os.path.join(CODE, "dual_agent_mc.csv")
    d.to_csv(output_path, index=False)
    print(f"\n  [✔] 结果已永久保存至: {output_path}")

if __name__ == "__main__":
    main()