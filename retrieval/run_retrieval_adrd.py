#!/usr/bin/env python3
import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import config
from advanced_retriever import AdvancedRetriever
from llm_utils import evaluate_context, rewrite_tf_query, evaluate_tf_evidence, decompose_mc_options

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
JSON_DIR = os.path.join(PROJECT_ROOT, "data")
MC_PATH  = os.path.join(JSON_DIR, "ADRD_Caregiving_Multiple_Choice.json")
TF_PATH  = os.path.join(JSON_DIR, "ADRD_Caregiving_True_or_False.json")

def load_adrd_bench_questions(subset="all"):
    records = []
    if subset in ("mc", "all") and os.path.exists(MC_PATH):
        with open(MC_PATH, encoding="utf-8") as f:
            for item in json.load(f)["data"]:
                options = item.get("Options", {})
                ans_letter = item["Answer"]
                query = f"{item['Question']}\nOptions:\n" + "\n".join([f"  {k}. {v}" for k, v in options.items()])
                records.append({"Question_ID": f"ADRD_MC_{item['ID']:03d}", "Type": "MC", "Question": query, "Stem": item["Question"], "Options_Dict": options, "Ground_Truth_Answer": options.get(ans_letter, ans_letter), "Correct_Letter": ans_letter})
    if subset in ("tf", "all") and os.path.exists(TF_PATH):
        with open(TF_PATH, encoding="utf-8") as f:
            for item in json.load(f)["data"]:
                records.append({"Question_ID": f"ADRD_TF_{item['ID']:03d}", "Type": "TF", "Question": item["Question"], "Ground_Truth_Answer": item["Answer"], "Correct_Letter": item["Answer"]})
    return records

def main():
    parser = argparse.ArgumentParser(description="Batch retrieval for ADRD-Bench (Local Precision Mode)")
    parser.add_argument("--top_k",  type=int,   default=config.retrieval_top_k, help="Final passages to return per question")
    parser.add_argument("--pre_k",  type=int,   default=config.retrieval_pre_k, help="Candidates before reranking")
    parser.add_argument("--window", type=int,   default=config.retrieval_window_size,  help="Context window expansion chars")
    parser.add_argument("--subset", choices=["mc", "tf", "all"], default=config.default_subset, help="Subset to retrieve")
    args = parser.parse_args()

    print("🔧 Initializing AdvancedRetriever (Local Only)...")
    retriever = AdvancedRetriever()
    questions = load_adrd_bench_questions(subset=args.subset)
    
    os.makedirs(os.path.join(PROJECT_ROOT, "retrieval_results"), exist_ok=True)
    output_path = os.path.join(PROJECT_ROOT, "retrieval_results", f"retrieval_ADRD_{args.subset}_CLEAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    results = []
    for item in tqdm(questions, desc="Retrieving"):
        q_type, original_question, question_id = item["Type"], item["Question"], item["Question_ID"]
        queries, retrieved_contexts, sources, scores, retrieved_chunk_ids = [], [], [], [], set()

        if q_type == "TF":
            try:
                rw = json.loads(rewrite_tf_query(original_question))
                queries = [q for q in rw.get("queries", []) if q != original_question] or [original_question]
            except: queries = [original_question]
        elif q_type == "MC":
            try:
                decomp = json.loads(decompose_mc_options(item.get("Stem", ""), item.get("Options_Dict", {})))
                queries = list(decomp.get("option_queries", {}).values()) or [original_question]
            except: queries = [original_question]

        for q in queries:
            passages, chunk_scores, chunk_sources = retriever.get_retrieved_passages(
                q, top_k=args.top_k, bm25_weight=config.bm25_weight, vector_weight=config.vector_weight, pre_k=args.pre_k, window_size=args.window
            )
            for p, s, src in zip(passages, chunk_scores, chunk_sources):
                pid = f"{src}__{hash(p)}"
                if pid not in retrieved_chunk_ids:
                    retrieved_contexts.append(p); sources.append(src); scores.append(s); retrieved_chunk_ids.add(pid)

        tf_verdict, satisfied = None, False
        if q_type == "TF" and retrieved_contexts:
            try:
                tf_eval = json.loads(evaluate_tf_evidence(original_question, retrieved_contexts))
                tf_verdict = tf_eval.get("verdict", "insufficient")
                satisfied = tf_verdict in ("True", "False")
            except: pass
        elif q_type == "MC" and retrieved_contexts:
            try:
                eval_res = json.loads(evaluate_context(original_question, retrieved_contexts))
                satisfied = eval_res.get("status") == "answerable"
            except: pass

        print(f"📄 [{question_id}] Final: {len(retrieved_contexts)} passages | satisfied={satisfied}" + (f" | tf_verdict={tf_verdict}" if q_type == 'TF' else ""))
              
        results.append({
            "Question_ID": question_id, "Type": q_type, "Question": original_question,
            "Ground_Truth_Answer": item["Ground_Truth_Answer"], "Correct_Letter": item["Correct_Letter"],
            "Retrieved_Passages": json.dumps(retrieved_contexts, ensure_ascii=False),
            "Retrieved_Sources": json.dumps(sources, ensure_ascii=False),
            "Satisfied": satisfied, "TF_Verdict": tf_verdict
        })

    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Local Precision Retrieval complete! Saved to {output_path}")

if __name__ == "__main__":
    main()