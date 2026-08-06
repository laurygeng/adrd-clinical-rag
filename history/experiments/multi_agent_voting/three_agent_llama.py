#!/usr/bin/env python3
"""3rd agent = local Llama-3.2-3B. Optimized with TQDM Progress Bar, Breakpoint Resume, and Memory Management."""
import os, sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- 定义你的项目根目录绝对路径 ---
ROOT_DIR = "/Users/minjie/Desktop/code/rag_adrd"
CODE_DIR = f"{ROOT_DIR}/code"
FINAL_OUT_PATH = f"{CODE_DIR}/three_agent_llama.csv"

print("--> 正在加载本地 Embedding 模型和 Llama 权重 (请保证系统内存充足)...", flush=True)
emb = SentenceTransformer("all-MiniLM-L6-v2")
M = "meta-llama/Llama-3.2-3B-Instruct"

TOKEN = open(f"{ROOT_DIR}/.secrets/hf_token.txt").read().strip()
tok = AutoTokenizer.from_pretrained(M, token=TOKEN)
tok.pad_token = tok.eos_token 

# 加载模型至 mps 加速
mdl = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, token=TOKEN).to("mps")
mdl.eval()

SYS_ID_MC=("You judge whether a CONTEXT is sufficient to answer a multiple-choice question. Name the SINGLE most "
  "important piece of information still MISSING needed to confidently determine the correct option. If everything "
  "needed is present, answer exactly 'NONE'. One short phrase only.")
SYS_ID_TF=("You judge whether a CONTEXT is sufficient to decide if a TRUE/FALSE statement is true or false. Name the "
  "SINGLE most important piece of information still MISSING needed to confidently decide. If everything needed is "
  "present, answer exactly 'NONE'. One short phrase only.")
SYS_V="Decide if a specific piece of information is PRESENT in the context. Answer exactly 'PRESENT' or 'ABSENT'."

def gen(sysp, user, temp, mx):
    msgs=[{"role":"system","content":sysp},{"role":"user","content":user}]
    ids=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("mps")
    am = torch.ones_like(ids)
    with torch.no_grad():
        out=mdl.generate(ids, attention_mask=am, max_new_tokens=mx, 
                         do_sample=(temp>0), temperature=max(temp, 1e-5), 
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def is_none(s): return s.strip().upper().startswith("NONE") or not s.strip()

def qwen_verdict(sysid, qb, ctx):
    gaps = []
    # 串行执行，避免 Mac 显存溢出
    for _ in range(3):
        gaps.append(gen(sysid, f"Context:\n{ctx}\n\n{qb}", 0.7, 40))
        
    nf=float(np.mean([is_none(g) for g in gaps])); real=[g for g in gaps if not is_none(g)]
    if nf>=0.6 or not real: return 1, nf
    
    E=emb.encode(real, normalize_embeddings=True); cons=real[int((E@E.T).mean(1).argmax())]
    
    v_ans = gen(SYS_V, f"Context:\n{ctx}\n\nInformation: {cons}\n\nPresent?", 0, 6)
    return (1,nf) if "PRESENT" in v_ans.upper() else (0, min(nf,0.3))

def auc(y,s):
    y=np.array(y).astype(int); s=np.array(s,float); P,N=s[y==1],s[y==0]
    return float((P[:,None]>N[None,:]).mean()+0.5*(P[:,None]==N[None,:]).mean()) if len(P) and len(N) else np.nan

def boot(d,col,B=2000):
    rng=np.random.default_rng(0); n=len(d); v=[]
    for _ in range(B):
        s=d.iloc[rng.integers(0,n,n)]
        if s.correct.nunique()>1: v.append(auc(s.correct,s[col]))
    return np.percentile(v,[2.5,97.5])

def bdiff(d,c1,c2,B=2000):
    rng=np.random.default_rng(0); n=len(d); v=[]
    for _ in range(B):
        s=d.iloc[rng.integers(0,n,n)]
        if s.correct.nunique()>1: v.append(auc(s.correct,s[c1])-auc(s.correct,s[c2]))
    v=np.array(v); return np.percentile(v,[2.5,97.5]),(v>0).mean()

def main():
    ret=pd.read_csv(f"{CODE_DIR}/retrieval_results/retrieval_ADRD_all_MERGED_corrected_bucketA.csv").set_index("Question_ID")
    mcj={f"ADRD_MC_{it['ID']:03d}": it for it in json.load(open(f"{CODE_DIR}/data/ADRD_Caregiving_Multiple_Choice.json"))["data"]}

    print("--> 正在载入 GPT-4 与 Gemini 的历史评估数据...")
    mc=pd.read_csv(f"{CODE_DIR}/dual_agent_mc.csv").rename(columns={"a5":"g4","b5":"gm"})[["qid","correct","g4","gm","sA","sB"]]
    tf=pd.read_csv(f"{CODE_DIR}/dual_agent_tf.csv").rename(columns={"sglA":"g4","sglB":"gm"})[["qid","correct","g4","gm","sA","sB"]]
    prev=pd.concat([mc,tf],ignore_index=True).set_index("qid")

    # 断点续传核心逻辑
    completed_qids = set()
    if os.path.exists(FINAL_OUT_PATH):
        try:
            existing_df = pd.read_csv(FINAL_OUT_PATH)
            if not existing_df.empty and "qid" in existing_df.columns:
                completed_qids = set(existing_df["qid"].dropna().tolist())
                print(f"--> [断点检测] 发现已有进度文件，将自动跳过已处理的 {len(completed_qids)} 道题。")
        except Exception:
            print("--> 进度文件损坏或为空，将重新开始完整计算。")

    print(f"--> 开始调度本地 Llama-3.2 (安全防溢出模式, 样本总数 N={len(prev)})...")
    
    for qid, r in tqdm(prev.iterrows(), total=len(prev), desc="Llama 审判进度"):
        if qid in completed_qids:
            continue
            
        row = ret.loc[qid]
        ctx = "\n\n".join(json.loads(row["Retrieved_Passages"])[:8])
        
        if qid.startswith("ADRD_MC"):
            it = mcj[qid]
            qb = "Question:\n"+it["Question"]+"\nOptions:\n"+"\n".join(f"{k}. {v}" for k,v in it["Options"].items())
            sysid = SYS_ID_MC
        else:
            qb = "Statement:\n"+str(row["Question"])
            sysid = SYS_ID_TF
            
        qv, qs = qwen_verdict(sysid, qb, ctx)
        
        new_row = pd.DataFrame([{
            "qid": qid, "correct": int(r["correct"]), "g4": r["g4"], "gm": r["gm"], "qw": qs,
            "sA": int(r["sA"]), "sB": int(r["sB"]), "sQ": qv
        }])
        
        # 实时追加保存
        if not os.path.exists(FINAL_OUT_PATH) or len(completed_qids) == 0:
            new_row.to_csv(FINAL_OUT_PATH, index=False)
            completed_qids.add(qid)
        else:
            new_row.to_csv(FINAL_OUT_PATH, mode='a', header=False, index=False)
            
        # ✨ 最关键的防崩溃代码：每做完一题，手动清空 Mac 显存垃圾
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n--> 所有题目计算完毕，正在做最终的统计分析...")
    d = pd.read_csv(FINAL_OUT_PATH)
    
    d["dual_vote"] = d.sA + d.sB + 0.01 * (d.g4 + d.gm)
    d["tri_avg"] = (d.g4 + d.gm + d.qw) / 3
    d["tri_wavg"] = 0.4 * d.g4 + 0.4 * d.gm + 0.2 * d.qw
    d["tri_vote"] = d.sA + d.sB + d.sQ + 0.001 * (d.g4 + d.gm + d.qw)
    
    d.to_csv(FINAL_OUT_PATH, index=False)

    print(f"\n===== 终极全样本比拼 POOLED TF+MC (n={len(d)}, 对/错={int(d.correct.sum())}/{int((1-d.correct).sum())}) =====")
    for c, nm in [("g4","gpt-4"), ("gm","gemini"), ("qw","Llama-3.2-3B solo"), ("dual_vote","2-agent vote (baseline)"),
                 ("tri_avg","3-agent equal-avg"), ("tri_wavg","3-agent weighted-avg"), ("tri_vote","3-agent vote")]:
        lo, hi = boot(d, c)
        print(f"  {nm:26s} AUC={auc(d.correct, d[c]):.3f} [95%CI {lo:.3f},{hi:.3f}]")
        
    print("\n===== 显著性差异检验 (Vs. 双智能体基准线) =====")
    for c in ["tri_avg", "tri_wavg", "tri_vote"]:
        (lo, hi), p = bdiff(d, c, "dual_vote")
        print(f"  {c} − 2agent: Δ AUC CI[{lo:+.3f},{hi:+.3f}] P(Δ>0)={p:.0%}")
        
    print(f"\n[✔] 最终分析结果已成功更新并保存至: {FINAL_OUT_PATH}")

if __name__ == "__main__":
    main()