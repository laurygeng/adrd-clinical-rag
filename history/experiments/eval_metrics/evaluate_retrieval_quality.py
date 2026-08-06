#!/usr/bin/env python3
"""
Retrieval-quality evaluator — measures the thing we actually care about for the
KB-gap auto-completion research: does the RETRIEVED CONTEXT contain the fact needed
to reach the ground-truth answer?  (present / partial / absent)

This is decoupled from generation: no GPT-4 answering, no accuracy, no 4-5h runs.
It is an LLM judge over a retrieval CSV, so a config change can be A/B'd in minutes.

Metric: fact-recall = #present / N   (and #(present+partial)/N as a soft recall).

Usage (from code/retrieval/):
  python evaluate_retrieval_quality.py --retrieval ../retrieval_results/<file>.csv
  python evaluate_retrieval_quality.py --retrieval <file>.csv --ids ADRD_TF_018,ADRD_MC_013
"""
import os, sys, json, argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_utils import get_openai_client, _chat_with_retry
from rag_config import config

JUDGE_SYS = (
    "You are a strict retrieval-quality judge for an ADRD medical/caregiving QA system. "
    "You are given a QUESTION, its known CORRECT ANSWER, and a set of RETRIEVED PASSAGES. "
    "Decide ONLY whether the passages CONTAIN the specific fact/evidence a reader would need "
    "to arrive at the correct answer. Do NOT use outside knowledge; judge the passages only.\n"
    "Output JSON: {\"support\": \"present\" | \"partial\" | \"absent\", \"evidence\": \"<quote or ''>\", \"reason\": \"...\"}\n"
    "- present  = a passage states the key discriminating fact (e.g. the specific number/definition) directly.\n"
    "- partial  = related/adjacent info is there but the exact discriminating fact is missing.\n"
    "- absent   = nothing in the passages supports the correct answer."
)

def judge_one(question, answer, passages, model):
    client = get_openai_client()
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    user = f"QUESTION:\n{question}\n\nCORRECT ANSWER:\n{answer}\n\nRETRIEVED PASSAGES:\n{ctx}"
    try:
        r = _chat_with_retry(client, model=model,
                             messages=[{"role": "system", "content": JUDGE_SYS},
                                       {"role": "user", "content": user}],
                             response_format={"type": "json_object"})
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {"support": "error", "reason": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--ids", default=None, help="comma-separated Question_IDs (default: all rows)")
    ap.add_argument("--model", default=config.llm_eval_model)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.retrieval)
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        df = df[df.Question_ID.isin(want)]
    rows = df.to_dict("records")

    def work(r):
        ps = json.loads(r["Retrieved_Passages"])
        v = judge_one(r["Question"], r.get("Ground_Truth_Answer", ""), ps, args.model)
        return r["Question_ID"], r["Type"], v.get("support", "error"), v.get("reason", "")[:80]

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, r) for r in rows]):
            results.append(fut.result())
    results.sort()

    from collections import Counter
    c = Counter(s for _, _, s, _ in results)
    n = len(results)
    present = c.get("present", 0); partial = c.get("partial", 0)
    print(f"\n=== Retrieval-quality on {n} questions ({os.path.basename(args.retrieval)}) ===")
    print(f"  present: {present}  partial: {partial}  absent: {c.get('absent',0)}  error: {c.get('error',0)}")
    print(f"  HARD fact-recall (present):        {present}/{n} = {present/n*100:.1f}%")
    print(f"  SOFT fact-recall (present+partial): {present+partial}/{n} = {(present+partial)/n*100:.1f}%")
    print("\n  absent/partial questions (gaps):")
    for qid, t, s, reason in results:
        if s in ("absent", "partial"):
            print(f"    {qid} [{t}] {s}: {reason}")

if __name__ == "__main__":
    main()
