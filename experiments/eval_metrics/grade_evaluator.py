#!/usr/bin/env python3
"""Make 'is the sufficiency evaluator any good?' CONCRETE.
Ground-truth proxy: answer-correct == context-was-sufficient.
Scores all 149 ADRD questions with the saved medical-grounded base model, thresholds
P(sufficient) at 0.5, and prints a confusion matrix + accuracy / precision / recall,
per type. No API, fully local."""
import json
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = "/tmp/suff_roberta_med"; MAXLEN = 256; THR = 0.5
tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForSequenceClassification.from_pretrained(BASE).eval()

ret = pd.read_csv("../retrieval_results/retrieval_ADRD_all_MERGED_corrected_bucketA.csv").set_index("Question_ID")
gen = pd.read_csv("../generate/answers/answers_gpt4_ADRD_all_rag_20260617_205522.csv").set_index("Question_ID")
ret = ret.join(gen["Is_Correct"]); ret["passages"] = ret["Retrieved_Passages"].apply(json.loads)

def p_sufficient(q, ctx):
    enc = tok(q, ctx, truncation=True, max_length=MAXLEN, return_tensors="pt")
    with torch.no_grad():
        return torch.softmax(model(**enc).logits, -1)[0, 1].item()

rows = []
for qid, r in ret.iterrows():
    p = p_sufficient(str(r["Question"]), "\n\n".join(r["passages"][:8]))
    rows.append({"type": r["Type"], "correct": bool(r["Is_Correct"]),
                 "pred_sufficient": p >= THR, "p": p})
df = pd.DataFrame(rows)

def report(d, name):
    TP = ((d.pred_sufficient) & (d.correct)).sum()       # said enough, answer right
    TN = ((~d.pred_sufficient) & (~d.correct)).sum()      # said NOT enough, answer wrong  (correctly flagged)
    FP = ((d.pred_sufficient) & (~d.correct)).sum()       # said enough, answer WRONG  (dangerous miss)
    FN = ((~d.pred_sufficient) & (d.correct)).sum()       # said NOT enough, answer right (false alarm)
    n = len(d); acc = (TP + TN) / n
    catch = TN / (TN + FP) if (TN + FP) else float("nan")  # of wrong-answer Qs, how many flagged
    print(f"\n[{name}]  n={n}  accuracy={acc:.3f}")
    print(f"   said-ENOUGH & right (TP)  = {TP:>3}     said-ENOUGH & WRONG (FP, missed) = {FP:>3}")
    print(f"   said-not-enough & right(FN)= {FN:>3}     said-not-enough & wrong (TN, caught)= {TN:>3}")
    print(f"   'catch rate' (flagged the wrong-answer Qs for补充) = {catch:.3f}")

for t, name in [("TF", "TF"), ("MC", "MC")]:
    report(df[df.type == t], name)
report(df, "ALL")
