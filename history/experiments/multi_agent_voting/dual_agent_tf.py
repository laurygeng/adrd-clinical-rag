#!/usr/bin/env python3
"""Dual-agent heterogeneous voting for sufficiency on TF (more negatives than MC).
Reuses dual_agent_sufficiency machinery; TF-specific identify prompt. Reports AUC + bootstrap,
then pools with the saved MC run for a combined estimate."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dual_agent_sufficiency as D

SYS_ID_TF = ("You judge whether a CONTEXT is sufficient to decide if a TRUE/FALSE statement is true or false. "
             "Name the SINGLE most important piece of information still MISSING from the context needed to confidently "
             "decide. If the context already contains everything needed, answer exactly 'NONE'. One short phrase only.")

def tf_identify(backend, model, stmt, ctx, n):
    return [D.call(backend, model, SYS_ID_TF, f"Context:\n{ctx}\n\nStatement:\n{stmt}", 0.7, 40) for _ in range(n)]

def boot_auc(d, col, B=2000, seed=0):
    rng=np.random.default_rng(seed); n=len(d); v=[]
    for _ in range(B):
        s=d.iloc[rng.integers(0,n,n)]
        if s.correct.nunique()<2: continue
        v.append(D.auc(s.correct,s[col]))
    return np.percentile(v,[2.5,50,97.5])
def boot_diff(d,c1,c2,B=2000,seed=0):
    rng=np.random.default_rng(seed); n=len(d); v=[]
    for _ in range(B):
        s=d.iloc[rng.integers(0,n,n)]
        if s.correct.nunique()<2: continue
        v.append(D.auc(s.correct,s[c1])-D.auc(s.correct,s[c2]))
    v=np.array(v); return np.percentile(v,[2.5,50,97.5]),(v>0).mean()

def main():
    ret = pd.read_csv(D.RET).set_index("Question_ID"); gen = pd.read_csv(D.GEN).set_index("Question_ID")
    ret = ret.join(gen["Is_Correct"]); rows=[]
    tf = ret[ret.Type=="TF"]
    for qid, r in tf.iterrows():
        stmt=str(r["Question"]); ctx="\n\n".join(json.loads(r["Retrieved_Passages"])[:8])
        gA=tf_identify(*D.AGENT_A, stmt, ctx, 3); gB=tf_identify(*D.AGENT_B, stmt, ctx, 3)
        sA,scA=D.verdict(gA, ctx, *D.AGENT_A); sB,scB=D.verdict(gB, ctx, *D.AGENT_B)
        rows.append({"qid":qid,"correct":int(bool(r["Is_Correct"])),"sglA":scA,"sglB":scB,
                     "dual_avg":(scA+scB)/2,"dual_min":min(scA,scB),"dual_vote":sA+sB+0.01*(scA+scB),"sA":sA,"sB":sB})
        print(f"  {qid}: gpt4={scA:.2f} gem={scB:.2f} vote={sA+sB} correct={int(bool(r['Is_Correct']))}", flush=True)
    d=pd.DataFrame(rows)
    
    # 修改 1：使用永久路径保存 tf 结果
    tf_out_path = os.path.join(D.CODE, "dual_agent_tf.csv")
    d.to_csv(tf_out_path, index=False)
    
    print(f"\n===== TF (n={len(d)}, 对/错={int(d.correct.sum())}/{int((1-d.correct).sum())}) =====")
    for c,nm in [("sglA","single gpt-4"),("sglB","single gemini"),("dual_vote","DUAL vote"),("dual_avg","DUAL avg")]:
        lo,md,hi=boot_auc(d,c); print(f"  {nm:14s} AUC={D.auc(d.correct,d[c]):.3f}  [95%CI {lo:.3f},{hi:.3f}]")
    (lo,md,hi),p=boot_diff(d,"dual_vote", d[["sglA","sglB"]].apply(lambda x: "sglA" if D.auc(d.correct,d.sglA)>=D.auc(d.correct,d.sglB) else "sglB",axis=1).iloc[0] if False else "sglA")
    print(f"  agreement(sA==sB): {(d.sA==d.sB).mean():.0%}")
    
    # pooled with MC
    try:
        # 修改 2：使用永久路径读取 mc 结果
        mc_in_path = os.path.join(D.CODE, "dual_agent_mc.csv")
        mc = pd.read_csv(mc_in_path).rename(columns={"a5":"sglA","b5":"sglB"})
        
        cols=["correct","sglA","sglB","dual_avg","dual_min","dual_vote"]
        pool=pd.concat([d[cols], mc[cols]], ignore_index=True)
        print(f"\n===== POOLED TF+MC (n={len(pool)}, 对/错={int(pool.correct.sum())}/{int((1-pool.correct).sum())}) =====")
        for c,nm in [("sglA","single gpt-4"),("sglB","single gemini"),("dual_vote","DUAL vote")]:
            lo,md,hi=boot_auc(pool,c); print(f"  {nm:14s} AUC={D.auc(pool.correct,pool[c]):.3f}  [95%CI {lo:.3f},{hi:.3f}]")
        best="sglB" if D.auc(pool.correct,pool.sglB)>=D.auc(pool.correct,pool.sglA) else "sglA"
        (lo,md,hi),pp=boot_diff(pool,"dual_vote",best)
        print(f"  DUAL vote − best single ({best}): Δ={md:+.3f} [95%CI {lo:+.3f},{hi:+.3f}] P(Δ>0)={pp:.0%}")
    except Exception as e:
        print("pool skipped:", repr(e)[:120])

if __name__=="__main__": main()