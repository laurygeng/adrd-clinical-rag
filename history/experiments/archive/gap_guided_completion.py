#!/usr/bin/env python3
"""
Step 3 (completion) — GAP-GUIDED LOCAL RE-RETRIEVAL.

Once the evaluator says a question is under-answered, we now have a precise notion of
WHAT is missing. Instead of searching the web, first re-query the LOCAL KB with that
specific gap: the answer-bearing passage is often already in the KB but was never
surfaced by the generic question. This is fully self-contained (no external API/search).

For each target question:
  1. Generate a targeted gap query (the specific fact needed) — 1 cheap LLM call.
  2. Re-retrieve from the local KB using that gap query.
  3. Merge gap-passages with the original top context (dedup).
  4. Re-generate the answer; report whether it flips vs the baseline.

Run on the failing questions (recovery) + some correct controls (regression check).
"""
import os, sys, json, argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai import OpenAI
from advanced_retriever import AdvancedRetriever

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RET = os.path.join(CODE, "retrieval_results", "retrieval_ADRD_all_MERGED_corrected_bucketA.csv")
GEN = os.path.join(CODE, "generate", "answers", "answers_gpt4_ADRD_all_rag_20260617_205522.csv")
MC_JSON = os.path.join(CODE, "data", "ADRD_Caregiving_Multiple_Choice.json")
sys.path.insert(0, os.path.join(CODE, "generate"))
from generate_answers_gpt4_ADRD_Bench import generate_answer, check_accuracy
client = OpenAI()
MINI = "gpt-4o-mini"

def gap_query(question):
    r = client.chat.completions.create(model=MINI, temperature=0.0, max_tokens=40,
        messages=[{"role": "system", "content":
            "Given a question, write ONE short search phrase (<=12 words) naming the SPECIFIC fact "
            "needed to answer it (a number, definition, recommendation, or discriminating detail). "
            "Output only the phrase."},
            {"role": "user", "content": question}])
    return r.choices[0].message.content.strip()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", default=None); args = ap.parse_args()
    ret = pd.read_csv(RET).set_index("Question_ID")
    gen = pd.read_csv(GEN).set_index("Question_ID")
    ret = ret.join(gen["Is_Correct"])
    mc_items = {f"ADRD_MC_{it['ID']:03d}": it for it in json.load(open(MC_JSON))["data"]}

    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    else:  # default: all failing questions
        ids = list(ret[ret.Is_Correct == False].index)
    ids = [q for q in ids if q in ret.index]

    print(f"Loading AdvancedRetriever for gap-guided local re-retrieval... ({len(ids)} targets)")
    retr = AdvancedRetriever()

    rows = []
    for qid in ids:
        r = ret.loc[qid]
        qtype = r["Type"]
        # question text for generation (MC needs options block)
        if qtype == "MC":
            it = mc_items[qid]
            qtext = it["Question"] + "\n\nOptions:\n" + "\n".join(f"  {k}. {v}" for k, v in it["Options"].items()) + "\n"
        else:
            qtext = str(r["Question"])

        orig = json.loads(r["Retrieved_Passages"])
        gq = gap_query(qtext)
        gap_passages, _, _ = retr.get_retrieved_passages(gq, top_k=8,
            bm25_weight=0.3, vector_weight=0.7, pre_k=30, window_size=800)
        # merge: original top-8 + gap-retrieved, dedup
        merged, seen = [], set()
        for p in orig[:8] + gap_passages:
            k = p.strip()[:120]
            if k not in seen:
                seen.add(k); merged.append(p)
        context = "\n\n".join(f"--- Snippet {i+1} ---\n{p}" for i, p in enumerate(merged[:12]))

        ans = generate_answer(client, qtext, context, qtype)
        ok = check_accuracy(ans, r["Ground_Truth_Answer"], r["Correct_Letter"], qtype)
        was = bool(r["Is_Correct"])
        flag = "↑RECOVER" if (ok and not was) else ("↓REGRESS" if (was and not ok) else "=")
        rows.append({"qid": qid, "type": qtype, "was_correct": was, "now_correct": ok, "flag": flag, "gap_query": gq})
        print(f"  {qid} [{qtype}] {was}->{ok} {flag}  gap='{gq[:50]}'", flush=True)

    d = pd.DataFrame(rows)
    rec = ((~d.was_correct) & (d.now_correct)).sum()
    reg = (d.was_correct & (~d.now_correct)).sum()
    print(f"\n=== gap-guided LOCAL re-retrieval: recovered {rec}, regressed {reg}, net {rec-reg} (of {len(d)} targets) ===")
    print("recovered:", list(d[(~d.was_correct) & (d.now_correct)].qid))

if __name__ == "__main__":
    main()
