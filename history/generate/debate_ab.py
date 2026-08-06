#!/usr/bin/env python3
"""A/B for the Verify/Debate Panel (agent ⑤) on TF questions:
single-pass baseline vs Proponent–Opponent–Judge debate, same retrieved context.
Usage: python debate_ab.py <retrieval_csv> <baseline_answers_csv> [--all|--sample N]"""
import pandas as pd, json, sys
import generate_answers_gpt4_ADRD_Bench as G
from debate_panel import debate_tf
from openai import OpenAI

def main():
    ret_csv = sys.argv[1]; base_csv = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "--all"
    r = pd.read_csv(ret_csv).set_index("Question_ID")
    g = pd.read_csv(base_csv).set_index("Question_ID")
    tf = g[g.Type == "TF"]
    ids = list(tf.index)
    if mode == "--sample":
        ids = ids[:int(sys.argv[4])]
    client = OpenAI()
    base_ok = deb_ok = 0; fixed = []; broke = []
    for qid in ids:
        row = r.loc[qid]; ctx = G.build_context_from_passages(json.loads(row["Retrieved_Passages"]), 10)
        stmt = str(row["Question"]); gold = row["Ground_Truth_Answer"]; letter = row["Correct_Letter"]
        base = G.generate_answer(client, stmt, ctx, "TF"); bcorr = bool(G.check_accuracy(base, gold, letter, "TF"))
        dv, _, _ = debate_tf(stmt.replace("True or False statement:", "").strip(), ctx)
        dcorr = bool(G.check_accuracy(dv, gold, letter, "TF"))
        base_ok += bcorr; deb_ok += dcorr
        if dcorr and not bcorr: fixed.append(qid)
        if bcorr and not dcorr: broke.append(qid)
        print(f"{qid}: gold={gold} base={'Y' if bcorr else 'N'} debate={'Y' if dcorr else 'N'}"
              + ("  <-- FIX" if (dcorr and not bcorr) else "  <-- BREAK" if (bcorr and not dcorr) else ""), flush=True)
    print(f"\nTF n={len(ids)}  Baseline {base_ok}  Debate {deb_ok}  net={deb_ok-base_ok:+d}", flush=True)
    print(f"fixed: {fixed}\nbroke: {broke}", flush=True)

if __name__ == "__main__":
    main()
