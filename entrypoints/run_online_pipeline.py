#!/usr/bin/env python3
"""
Online Pipeline Batch Runner
Role: Handles two distinct execution modes:
1. Benchmark Mode (JSONs): Evaluates ADRD MC/TF questions, calculates accuracy, and prints a report.
2. Inference Mode (CSV): Processes a custom dataset (e.g., 500 QA questions), appends the generated 
   answers and retrieved contexts as new columns to the original CSV, and exports it for later quality evaluation.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Ensure project root is in the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.orchestrator import run_pipeline
from core.answer_agent import check_accuracy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_benchmarks(subset="all"):
    """Loads MC and TF benchmark records from JSON files for accuracy evaluation."""
    records = []
    data_dir = os.path.join(PROJECT_ROOT, "data")
    mc_path = os.path.join(data_dir, "ADRD_Caregiving_Multiple_Choice.json")
    tf_path = os.path.join(data_dir, "ADRD_Caregiving_True_or_False.json")

    # Load Multiple Choice JSON
    if subset in ("mc", "all") and os.path.exists(mc_path):
        with open(mc_path, encoding="utf-8") as f:
            mc_data = json.load(f)
        for item in mc_data["data"]:
            q_id = f"ADRD_MC_{item['ID']:03d}"
            q_text = item["Question"]
            options = item.get("Options", {})
            ans_letter = item["Answer"]
            ground_truth_text = options.get(ans_letter, ans_letter)
            options_str = "\n".join([f"  {k}. {v}" for k, v in options.items()])
            formatted_q = f"{q_text}\n\nOptions:\n{options_str}\n"
            
            records.append({
                "Question_ID": q_id,
                "Type": "MC",
                "Question": formatted_q,
                "Ground_Truth_Answer": ground_truth_text,
                "Correct_Letter": ans_letter,
            })

    # Load True/False JSON
    if subset in ("tf", "all") and os.path.exists(tf_path):
        with open(tf_path, encoding="utf-8") as f:
            tf_data = json.load(f)
        for item in tf_data["data"]:
            q_id = f"ADRD_TF_{item['ID']:03d}"
            q_text = item["Question"]
            ground_truth = item["Answer"]
            formatted_q = f"True or False statement: {q_text}\n"
            
            records.append({
                "Question_ID": q_id,
                "Type": "TF",
                "Question": formatted_q,
                "Ground_Truth_Answer": ground_truth,
                "Correct_Letter": ground_truth,
            })

    return pd.DataFrame(records)

def run_benchmark_mode(subset, limit):
    """Executes the pipeline on JSON datasets and generates an Accuracy Report."""
    df_questions = load_json_benchmarks(subset)
    if df_questions.empty:
        logging.error("No JSON benchmark questions found in data/ folder.")
        return

    if limit > 0:
        df_questions = df_questions.head(limit)

    output_dir = os.path.join(PROJECT_ROOT, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"benchmark_eval_{subset}_{timestamp}.csv")

    print(f"\n🚀 [BENCHMARK MODE] Evaluating {len(df_questions)} questions...")
    
    results = []
    checkpoint_every = 10

    for idx, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="Running Benchmarks"):
        qid = row["Question_ID"]
        q_type = row["Type"]
        
        try:
            pipeline_output = run_pipeline(question=row["Question"], q_type=q_type)
            generated_answer = pipeline_output["final_answer"]
            
            is_correct = check_accuracy(
                generated=generated_answer,
                ground_truth=row["Ground_Truth_Answer"],
                correct_letter=row["Correct_Letter"],
                q_type=q_type
            )
        except Exception as e:
            logging.error(f"Failed on {qid}: {e}")
            generated_answer = f"ERROR: {e}"
            is_correct = False
            pipeline_output = {"trace": {}}

        results.append({
            "Question_ID": qid,
            "Type": q_type,
            "Question": row["Question"],
            "Generated_Answer": generated_answer,
            "Ground_Truth_Answer": row["Ground_Truth_Answer"],
            "Is_Correct": is_correct,
            "Veto_Triggered": pipeline_output["trace"].get("veto_triggered", False),
        })

        # Save checkpoint
        if (idx + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Final save
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Print Accuracy Report
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK ACCURACY REPORT")
    print(f"{'='*60}")
    for q_type in out_df["Type"].unique():
        sub = out_df[out_df["Type"] == q_type]
        acc = sub["Is_Correct"].mean() * 100
        print(f"  Type {q_type} ({len(sub)} Qs): {sub['Is_Correct'].sum()}/{len(sub)} = {acc:.1f}%")
    
    total_acc = out_df["Is_Correct"].mean() * 100
    print(f"  Overall Accuracy: {out_df['Is_Correct'].sum()}/{len(out_df)} = {total_acc:.1f}%")
    print(f"{'='*60}")
    print(f"✅ Report saved to: {output_csv}\n")


def run_inference_mode(csv_path, limit):
    """Executes the pipeline on a custom CSV, appending generated answers as new columns."""
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return

    df_original = pd.read_csv(csv_path)
    if limit > 0:
        df_original = df_original.head(limit)

    output_dir = os.path.join(PROJECT_ROOT, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(csv_path).replace(".csv", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"{filename}_answered_{timestamp}.csv")

    print(f"\n🚀 [INFERENCE MODE] Processing {len(df_original)} questions from custom CSV...")
    
    # 🌟 精准锁定问题列：优先完全匹配 "Question" 或 "Question - Reviewed"，且坚决避开 "ID"
    question_col = None
    for col in df_original.columns:
        clean_col = col.strip().lower()
        if clean_col == "question" or clean_col == "question - reviewed":
            question_col = col
            break
    
    # 兜底逻辑：如果没找到精确匹配，找包含 question 但不包含 id 的列
    if not question_col:
        for col in df_original.columns:
            if "question" in col.lower() and "id" not in col.lower():
                question_col = col
                break

    if not question_col:
        logging.error("❌ Could not find a valid 'Question' column in the CSV (avoiding 'Question ID').")
        return

    generated_answers = []
    retrieved_contexts = []
    veto_flags = []

    checkpoint_every = 10

    for idx, row in tqdm(df_original.iterrows(), total=len(df_original), desc="Generating Answers"):
        question_text = str(row[question_col])
        
        # Default to "QA" (Open-ended) if type is not strictly defined as MC or TF
        q_type = str(row.get("Type", row.get("type", "QA"))).upper()
        if q_type not in ["MC", "TF"]:
            q_type = "QA"

        try:
            pipeline_output = run_pipeline(question=question_text, q_type=q_type)
            generated_answers.append(pipeline_output["final_answer"])
            retrieved_contexts.append(pipeline_output["final_context"])
            veto_flags.append(pipeline_output["trace"].get("veto_triggered", False))
        except Exception as e:
            logging.error(f"Failed on row {idx}: {e}")
            generated_answers.append(f"ERROR: {e}")
            retrieved_contexts.append("")
            veto_flags.append(False)

        # Save checkpoint iteratively by modifying a copy of the dataframe
        if (idx + 1) % checkpoint_every == 0:
            df_temp = df_original.iloc[:idx+1].copy()
            df_temp["Generated_Answer"] = generated_answers
            df_temp["Final_Retrieved_Context"] = retrieved_contexts
            df_temp["Veto_Triggered"] = veto_flags
            df_temp.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Final save
    df_original["Generated_Answer"] = generated_answers
    df_original["Final_Retrieved_Context"] = retrieved_contexts
    df_original["Veto_Triggered"] = veto_flags
    df_original.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ Processing complete! New CSV generated successfully.")
    print(f"📁 Output saved to: {output_csv}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Online RAG Pipeline")
    parser.add_argument("--subset", choices=["mc", "tf", "all"], default="all", help="JSON dataset subset to run (Benchmark Mode)")
    parser.add_argument("--csv", type=str, default=None, help="Path to a custom CSV file (Inference Mode)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of questions to process")
    args = parser.parse_args()

    # =========================================================================
    # 🌟 严格的 API Key 前置预检机制（核心亮点保护）
    # =========================================================================
    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    
    missing_keys = []
    if not openai_key:
        missing_keys.append("OPENAI_API_KEY")
    if not google_key:
        missing_keys.append("GOOGLE_API_KEY")
        
    if missing_keys:
        print("\n" + "="*70)
        print("❌ [CRITICAL ERROR] Dual-Agent Voting System Pre-flight Check Failed!")
        print("="*70)
        print("Our system relies on a dual-agent heterogeneous voting mechanism (OpenAI + Gemini)")
        print("to evaluate context sufficiency. The following required API keys are missing:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease set both environment variables before starting the pipeline:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export GOOGLE_API_KEY='your_google_key'")
        print("="*70 + "\n")
        sys.exit(1)
    # =========================================================================

    # Routing logic
    if args.csv:
        run_inference_mode(args.csv, args.limit)
    else:
        run_benchmark_mode(args.subset, args.limit)

if __name__ == "__main__":
    main()