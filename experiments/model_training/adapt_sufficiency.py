#!/usr/bin/env python3
"""Stage 2 — ItV-distilled DOMAIN ADAPTATION.
Take the stage-1 general-pretrained RoBERTa, fine-tune it on ADRD sufficiency data
whose labels are DISTILLED FROM ItV (the validated teacher), then evaluate on a
held-out split of ADRD questions. Goal: lift zero-shot 0.53 toward ItV's ~0.73,
as a cheap one-forward-pass trained model (the core novelty).

Clean eval: split by QUESTION (no leakage). Compare trained-model vs ItV vs
stage-1(general-only) on the SAME held-out test questions.
"""
import os, json, random
import numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
import gate  # ItV teacher (tf_itv_sufficient, mc_sufficient)

random.seed(0)
STAGE1 = "/tmp/suff_roberta_med"   # medical-grounded base (general+PubMedQA): MC 0.69 / TF 0.58 zero-shot
OUT = "/tmp/suff_roberta_med_adapted"
MAXLEN = 256
dev = "cpu"

ret = pd.read_csv("../retrieval_results/retrieval_ADRD_all_MERGED_corrected_bucketA.csv").set_index("Question_ID")
gen = pd.read_csv("../generate/answers/answers_gpt4_ADRD_all_rag_20260617_205522.csv").set_index("Question_ID")
ret = ret.join(gen["Is_Correct"])
ret["passages"] = ret["Retrieved_Passages"].apply(json.loads)

ids = list(ret.index)
random.shuffle(ids)
test_ids = set(ids[:49]); train_ids = [q for q in ids if q not in test_ids]
print(f"split: train {len(train_ids)} / test {len(test_ids)} questions", flush=True)

def itv_label(q_type, question, ctx):
    if q_type == "TF":
        return int(gate.tf_itv_sufficient(question, ctx)[0])
    return int(gate.mc_sufficient(question, ctx)[0])

# ---- build ItV-distilled training data from TRAIN questions ----
print("Labeling training data with ItV teacher...", flush=True)
rows = []
for n, q in enumerate(train_ids):
    r = ret.loc[q]; ps = r["passages"]; qt = r["Type"]; question = str(r["Question"])
    full = "\n\n".join(ps[:8])
    deg = "\n\n".join(random.sample(ps[:8], k=min(2, len(ps[:8]))))          # degraded subset
    other = ret.loc[random.choice([x for x in train_ids if x != q])]["passages"]
    swap = "\n\n".join(other[:8])                                            # off-topic = insufficient
    rows.append({"question": question, "context": full, "label": itv_label(qt, question, full)})
    rows.append({"question": question, "context": deg,  "label": itv_label(qt, question, deg)})
    rows.append({"question": question, "context": swap, "label": 0})
    if (n + 1) % 20 == 0: print(f"  labeled {n+1}/{len(train_ids)}", flush=True)
ds = Dataset.from_list(rows)
print(f"train examples: {len(rows)} | sufficient={sum(x['label'] for x in rows)}", flush=True)

# ---- fine-tune stage-1 model ----
tok = AutoTokenizer.from_pretrained(STAGE1)
ds = ds.map(lambda b: tok(b["question"], b["context"], truncation=True, max_length=MAXLEN), batched=True)
model = AutoModelForSequenceClassification.from_pretrained(STAGE1, num_labels=2)
args = TrainingArguments(output_dir=OUT, num_train_epochs=3, use_cpu=True, per_device_train_batch_size=16,
                         learning_rate=1e-5, logging_steps=50, save_strategy="no", report_to=[])
Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorWithPadding(tok)).train()
model.to(dev).eval()
print("adapted + trained.", flush=True)

# ---- evaluate on held-out TEST questions ----
def model_score(question, ctx):
    enc = tok(question, ctx, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
    with torch.no_grad():
        return torch.softmax(model(**enc).logits, -1)[0, 1].item()

stage1_model = AutoModelForSequenceClassification.from_pretrained(STAGE1, num_labels=2).to(dev).eval()
stage1_tok = AutoTokenizer.from_pretrained(STAGE1)
def stage1_score(question, ctx):
    enc = stage1_tok(question, ctx, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
    with torch.no_grad():
        return torch.softmax(stage1_model(**enc).logits, -1)[0, 1].item()

def auc(y, s):
    y = np.array(y).astype(int); s = np.array(s, float); P, N = s[y == 1], s[y == 0]
    return float((P[:, None] > N[None, :]).mean() + 0.5 * (P[:, None] == N[None, :]).mean()) if len(P) and len(N) else float("nan")

evals = []
print("Evaluating on held-out test questions...", flush=True)
for q in test_ids:
    r = ret.loc[q]; ctx = "\n\n".join(r["passages"][:8]); question = str(r["Question"])
    evals.append({"correct": int(bool(r["Is_Correct"])),
                  "adapted": model_score(question, ctx),
                  "stage1": stage1_score(question, ctx),
                  "itv": float(itv_label(r["Type"], question, ctx))})
e = pd.DataFrame(evals)
print(f"\n=== held-out test (n={len(e)}) — AUC(sufficiency -> answer-correct) ===")
print(f"  stage-1 (general only) : {auc(e.correct, e.stage1):.3f}")
print(f"  ItV teacher (prompt)   : {auc(e.correct, e.itv):.3f}")
print(f"  ADAPTED (our trained)  : {auc(e.correct, e.adapted):.3f}")
print("  (target: adapted should beat stage-1's ~0.53 and approach ItV's ~0.73)")
