#!/usr/bin/env python3
"""Step 1 of the distilled answerability evaluator:
Train a small DeBERTa sufficiency classifier on GENERAL public data with natural
sufficiency labels (SQuAD 2.0 answerable/unanswerable [+ FEVER NEI if available]),
then ZERO-SHOT test it on ADRD — comparing AUC to CRAG (0.47) and our ItV (0.69-0.73).
Input = (question, context) -> P(sufficient). Fully local, no API."""
import os, json, random
import numpy as np, pandas as pd, torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

random.seed(0)
BASE = "roberta-base"          # mps-friendly + fast (DeBERTa-v3's disentangled attn OOM'd on mps)
OUT = "/tmp/suff_roberta_med"  # medical-grounded variant (keep original /tmp/suff_roberta for A/B)
MAXLEN = 256
N_PER_CLASS = 4000

def build_squad():
    sq = load_dataset("squad_v2", split="train").shuffle(seed=0)
    suff, insuff = [], []
    for ex in sq:
        rec = {"question": ex["question"], "context": ex["context"]}
        if len(ex["answers"]["text"]) > 0:
            if len(suff) < N_PER_CLASS: suff.append({**rec, "label": 1})
        else:
            if len(insuff) < N_PER_CLASS: insuff.append({**rec, "label": 0})
        if len(suff) >= N_PER_CLASS and len(insuff) >= N_PER_CLASS: break
    print(f"SQuAD2: sufficient={len(suff)} insufficient={len(insuff)}", flush=True)
    return suff + insuff

def build_fever():
    try:
        fv = load_dataset("copenlu/fever_gold_evidence", split="train").shuffle(seed=0)
        rows = []
        cap = N_PER_CLASS // 2
        c1 = c0 = 0
        for ex in fv:
            ev = ex.get("evidence");  lab = str(ex.get("label", "")).upper()
            ev = " ".join([e if isinstance(e, str) else " ".join(map(str, e)) for e in ev]) if isinstance(ev, list) else str(ev)
            if "NOT ENOUGH" in lab or lab == "NEI":
                if c0 < cap: rows.append({"question": ex["claim"], "context": ev, "label": 0}); c0 += 1
            elif lab in ("SUPPORTS", "REFUTES"):
                if c1 < cap: rows.append({"question": ex["claim"], "context": ev, "label": 1}); c1 += 1
            if c1 >= cap and c0 >= cap: break
        print(f"FEVER: sufficient={c1} insufficient={c0}", flush=True)
        return rows
    except Exception as e:
        print(f"FEVER skipped ({e})", flush=True)
        return []

def build_pubmedqa():
    """Medical-domain sufficiency from PubMedQA: each question's REAL abstract = sufficient,
    a swapped (unrelated) abstract = insufficient. Gives the model in-domain biomedical
    'is this context enough to answer' grounding that SQuAD2/FEVER (general) lack."""
    cap = N_PER_CLASS
    items = []
    try:
        pq = load_dataset("pubmed_qa", "pqa_artificial", split="train", streaming=True)
        for ex in pq:
            ctx = ex.get("context")
            if isinstance(ctx, dict):
                ctx = " ".join(ctx.get("contexts", []) or [])
            ctx = str(ctx).strip(); q = str(ex.get("question", "")).strip()
            if q and ctx:
                items.append((q, ctx))
            if len(items) >= cap:
                break
    except Exception as e:
        print(f"PubMedQA skipped ({e})", flush=True)
        return []
    rows, n = [], len(items)
    for i, (q, ctx) in enumerate(items):
        rows.append({"question": q, "context": ctx, "label": 1})                     # real abstract = sufficient
        rows.append({"question": q, "context": items[(i + n // 2) % n][1], "label": 0})  # swapped = insufficient
    print(f"PubMedQA: sufficient={n} insufficient={n}", flush=True)
    return rows

def main():
    data = build_squad() + build_fever() + build_pubmedqa()
    random.shuffle(data)
    ds = Dataset.from_list(data)
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = ds.map(lambda b: tok(b["question"], b["context"], truncation=True, max_length=MAXLEN), batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
    args = TrainingArguments(output_dir=OUT, num_train_epochs=1, use_cpu=True, per_device_train_batch_size=16,
                             learning_rate=2e-5, logging_steps=200, save_strategy="no", report_to=[])
    Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorWithPadding(tok)).train()
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("trained + saved.", flush=True)

    # ----- ZERO-SHOT on ADRD -----
    dev = "cpu"
    model.to(dev).eval()
    ret = pd.read_csv("../retrieval_results/retrieval_ADRD_all_MERGED_corrected_bucketA.csv").set_index("Question_ID")
    gen = pd.read_csv("../generate/answers/answers_gpt4_ADRD_all_rag_20260617_205522.csv").set_index("Question_ID")
    ret = ret.join(gen["Is_Correct"])
    def suff_score(q, passages):
        ctx = "\n\n".join(passages[:8])
        enc = tok(q, ctx, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, -1)[0, 1].item()  # P(sufficient)
        return p
    def auc(y, s):
        y = np.array(y).astype(int); s = np.array(s, float); P, N = s[y == 1], s[y == 0]
        return float((P[:, None] > N[None, :]).mean() + 0.5 * (P[:, None] == N[None, :]).mean()) if len(P) and len(N) else float("nan")
    rows = []
    for qid, r in ret.iterrows():
        rows.append({"type": r["Type"], "correct": int(bool(r["Is_Correct"])),
                     "score": suff_score(str(r["Question"]), json.loads(r["Retrieved_Passages"]))})
    d = pd.DataFrame(rows)
    print("\n=== trained-on-SQuAD2/FEVER/PubMedQA, ZERO-SHOT on ADRD ===")
    for t in ["TF", "MC"]:
        s = d[d.type == t]; print(f"  {t}: AUC = {auc(s.correct, s.score):.3f}")
    print(f"  ALL: AUC = {auc(d.correct, d.score):.3f}")
    print("  (general-only stage-1 was 0.53; CRAG 0.47; our ItV TF 0.694 / MC 0.731)")

if __name__ == "__main__":
    main()
