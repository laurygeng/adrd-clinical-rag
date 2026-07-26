#!/usr/bin/env python3
"""
Innovation #1 validation — NLI-based ANSWERABILITY evaluator vs the prompt-GPT-4o
high/med/low confidence.

Thesis: replace "relevance/similarity" with "answerability/entailment". A local,
offline NLI model scores, for each retrieved passage, whether it ENTAILS or
CONTRADICTS the claim (TF) / option (MC). The answerability signal = how decisively
the context supports an answer. We test whether this signal predicts answer
correctness BETTER than GPT-4o's confidence — fully local, no API.

Outputs:
  - AUC(signal -> Is_Correct) for NLI-margin vs GPT-4o confidence (higher = better gate)
  - NLI used directly as a zero-shot answerer (TF Yes/No, MC letter) accuracy
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentence_transformers import CrossEncoder

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RET = os.path.join(CODE, "retrieval_results", "retrieval_ADRD_all_MERGED_corrected_bucketA.csv")
GEN = os.path.join(CODE, "generate", "answers", "answers_gpt4_ADRD_all_rag_20260617_205522.csv")
MC_JSON = os.path.join(CODE, "data", "ADRD_Caregiving_Multiple_Choice.json")

def norm_tf(x):
    x = str(x).strip().lower()
    return "Yes" if x in ("yes", "true") else ("No" if x in ("no", "false") else "?")

def manual_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    # probability a random positive scores higher than a random negative
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

def main():
    print("Loading NLI model (local, offline after first download)...")
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    id2label = {int(k): v.lower() for k, v in nli.model.config.id2label.items()}
    lab2i = {v: k for k, v in id2label.items()}
    ENT, CON = lab2i["entailment"], lab2i["contradiction"]
    print("label map:", id2label)

    ret = pd.read_csv(RET).set_index("Question_ID")
    gen = pd.read_csv(GEN).set_index("Question_ID")
    ret = ret.join(gen["Is_Correct"])
    mc_items = {f"ADRD_MC_{it['ID']:03d}": it for it in json.load(open(MC_JSON))["data"]}

    def nli_probs(premises, hypothesis):
        if not premises: return np.zeros((1, 3))
        pairs = [(p, hypothesis) for p in premises]
        return np.asarray(nli.predict(pairs, apply_softmax=True, batch_size=32, show_progress_bar=False))

    # ---------- TF ----------
    tf_rows = []
    tf = ret[ret.Type == "TF"]
    for qid, r in tf.iterrows():
        ps = json.loads(r["Retrieved_Passages"])[:12]
        pr = nli_probs(ps, str(r["Question"]))
        support = pr[:, ENT].max(); refute = pr[:, CON].max()
        ans = "Yes" if support >= refute else "No"
        tf_rows.append({
            "gpt_conf": str(r.get("TF_Confidence")).lower(),
            "is_correct": int(bool(r["Is_Correct"])),
            "nli_answerability": float(max(support, refute)),   # how decisively context supports SOME answer
            "nli_decisiveness": float(abs(support - refute)),   # how cleanly it favors one side
            "nli_correct": int(norm_tf(ans) == norm_tf(r["Ground_Truth_Answer"])),
        })
    tfd = pd.DataFrame(tf_rows)
    tfd["gpt_num"] = tfd["gpt_conf"].map({"high": 2, "medium": 1, "low": 0}).fillna(0)

    print("\n===== TF (n=%d) : which signal predicts answer-correctness? (AUC, higher=better) =====" % len(tfd))
    print(f"  GPT-4o confidence (high/med/low) -> correct : AUC = {manual_auc(tfd.is_correct, tfd.gpt_num):.3f}")
    print(f"  NLI answerability                -> correct : AUC = {manual_auc(tfd.is_correct, tfd.nli_answerability):.3f}")
    print(f"  NLI decisiveness                 -> correct : AUC = {manual_auc(tfd.is_correct, tfd.nli_decisiveness):.3f}")
    print(f"  [bonus] NLI as zero-shot TF answerer accuracy : {tfd.nli_correct.mean()*100:.1f}%  (GPT-4o was {tfd.is_correct.mean()*100:.1f}%)")

    # ---------- MC ----------
    mc_rows = []
    mc = ret[ret.Type == "MC"]
    for qid, r in mc.iterrows():
        item = mc_items.get(qid)
        if not item: continue
        ps = json.loads(r["Retrieved_Passages"])[:12]
        opts = item["Options"]
        ent_by_opt = {}
        for k, v in opts.items():
            pr = nli_probs(ps, f"{item['Question']} {v}")
            ent_by_opt[k] = float(pr[:, ENT].max())
        order = sorted(ent_by_opt, key=ent_by_opt.get, reverse=True)
        top1, top2 = ent_by_opt[order[0]], (ent_by_opt[order[1]] if len(order) > 1 else 0.0)
        mc_rows.append({
            "is_correct": int(bool(r["Is_Correct"])),
            "nli_margin": float(top1 - top2),   # confidence = gap between best and runner-up option
            "nli_correct": int(order[0] == item["Answer"]),
        })
    mcd = pd.DataFrame(mc_rows)
    print("\n===== MC (n=%d) =====" % len(mcd))
    print(f"  NLI margin -> correct : AUC = {manual_auc(mcd.is_correct, mcd.nli_margin):.3f}")
    print(f"  [bonus] NLI as zero-shot MC answerer accuracy : {mcd.nli_correct.mean()*100:.1f}%  (GPT-4o was {mcd.is_correct.mean()*100:.1f}%)")

    print("\nInterpretation: if NLI-answerability AUC > GPT-4o-confidence AUC, the local NLI")
    print("signal is a better (and calibrated, API-free) sufficiency gate.")

if __name__ == "__main__":
    main()
