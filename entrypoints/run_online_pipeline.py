#!/usr/bin/env python3
"""
Online Pipeline Batch Runner (entrypoints/)
Role:
1) Benchmark Mode (JSONs): Evaluate ADRD MC/TF, calculate accuracy, export clean CSV with Retrieved_Context and timings. (Resumable)
2) Inference Mode (CSV): Process custom CSV, append answers. (Resumable)

** ABLATION SUPPORT ADDED **
Supports --no-rag and --no-completion flags.
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Optional, Set, List, Dict, Any

import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Path setup: entrypoints/ is sibling of core/ and data/
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core"))

from core.orchestrator import run_pipeline
from core.answer_agent import check_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _parse_ids(ids_str: Optional[str]) -> Optional[Set[int]]:
    if not ids_str:
        return None
    parts = [p.strip() for p in ids_str.split(",") if p.strip()]
    out: Set[int] = set()
    for p in parts:
        try:
            out.add(int(p))
        except ValueError:
            raise ValueError(f"--ids must be comma-separated integers. Bad token: '{p}'")
    return out


def load_json_benchmarks(subset: str = "all", allowed_ids: Optional[Set[int]] = None) -> pd.DataFrame:
    records: List[dict] = []
    data_dir = os.path.join(PROJECT_ROOT, "data")
    mc_path = os.path.join(data_dir, "ADRD_Caregiving_Multiple_Choice.json")
    tf_path = os.path.join(data_dir, "ADRD_Caregiving_True_or_False.json")

    if subset in ("mc", "all") and os.path.exists(mc_path):
        with open(mc_path, encoding="utf-8") as f:
            mc_data = json.load(f)
        for item in mc_data.get("data", []):
            q_num = int(item.get("ID"))
            if allowed_ids is not None and q_num not in allowed_ids:
                continue

            q_id = f"ADRD_MC_{q_num:03d}"
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
                "Numeric_ID": q_num,
            })

    if subset in ("tf", "all") and os.path.exists(tf_path):
        with open(tf_path, encoding="utf-8") as f:
            tf_data = json.load(f)
        for item in tf_data.get("data", []):
            q_num = int(item.get("ID"))
            if allowed_ids is not None and q_num not in allowed_ids:
                continue

            q_id = f"ADRD_TF_{q_num:03d}"
            q_text = item["Question"]
            ground_truth = item["Answer"]
            formatted_q = f"True or False statement: {q_text}\n"

            records.append({
                "Question_ID": q_id,
                "Type": "TF",
                "Question": formatted_q,
                "Ground_Truth_Answer": ground_truth,
                "Correct_Letter": ground_truth,
                "Numeric_ID": q_num,
            })

    df = pd.DataFrame(records)
    if not df.empty and "Numeric_ID" in df.columns:
        df = df.sort_values(["Type", "Numeric_ID"], ascending=[True, True]).reset_index(drop=True)
    return df


def run_benchmark_mode(subset: str, limit: int, ids_str: Optional[str], use_rag: bool, use_completion: bool):
    allowed_ids = _parse_ids(ids_str)
    df_questions = load_json_benchmarks(subset=subset, allowed_ids=allowed_ids)
    if df_questions.empty:
        logging.error("No JSON benchmark questions found (or none matched --ids).")
        return

    if limit > 0:
        df_questions = df_questions.head(limit)

    output_dir = os.path.join(PROJECT_ROOT, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    # 动态生成消融实验的文件标签
    ablation_tag = ""
    if not use_rag:
        ablation_tag += "_NO_RAG"
    if not use_completion:
        ablation_tag += "_NO_COMPLETION"

    prefix = f"benchmark_eval_{subset}{ablation_tag}_"
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".csv")]
    
    results = []
    processed_ids = set()
    
    if existing_files:
        existing_files.sort()
        output_csv = os.path.join(output_dir, existing_files[-1])
        print(f"\n🔄 [RESUME MODE] Found existing report: {output_csv}")
        try:
            existing_df = pd.read_csv(output_csv)
            if "Question_ID" in existing_df.columns:
                processed_ids = set(existing_df["Question_ID"].astype(str).tolist())
                results = existing_df.to_dict(orient="records")
                print(f"   Already processed {len(processed_ids)} questions. Resuming...")
        except Exception as e:
            logging.warning(f"Could not load existing CSV for resume: {e}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ids_tag = "" if allowed_ids is None else f"_ids_{','.join(str(x) for x in sorted(allowed_ids)).replace(',', '-')}"
            output_csv = os.path.join(output_dir, f"{prefix}{timestamp}{ids_tag}.csv")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ids_tag = "" if allowed_ids is None else f"_ids_{','.join(str(x) for x in sorted(allowed_ids)).replace(',', '-')}"
        output_csv = os.path.join(output_dir, f"{prefix}{timestamp}{ids_tag}.csv")

    remaining_df = df_questions[~df_questions["Question_ID"].astype(str).isin(processed_ids)].reset_index(drop=True)

    if remaining_df.empty:
        print("\n✅ All selected questions have already been processed in the latest CSV!")
        out_df = pd.DataFrame(results)
    else:
        print(f"\n🚀 [BENCHMARK MODE] Evaluating {len(remaining_df)} remaining questions (Sequential & Safe)...")
        print(f"🔧 Ablation Settings: Use RAG = {use_rag}, Use Completion = {use_completion}")
        checkpoint_every = 5

        for idx, row in tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Running Benchmarks"):
            qid = row["Question_ID"]
            q_type = row["Type"]
            question_text = row["Question"]

            t_start = time.time()
            retrieved_context = ""
            t_base, t_critic, t_comp, t_ans = None, None, None, None

            try:
                pipeline_output = run_pipeline(
                    question=question_text, 
                    q_type=q_type, 
                    question_id=qid,
                    use_rag=use_rag,
                    use_completion=use_completion
                )
                generated_answer = pipeline_output.get("final_answer", "")
                retrieved_context = pipeline_output.get("final_context", "")
                trace_dict = pipeline_output.get("trace", {})

                is_correct = check_accuracy(
                    generated=generated_answer,
                    ground_truth=row["Ground_Truth_Answer"],
                    correct_letter=row["Correct_Letter"],
                    q_type=q_type,
                )
                
                t_base = trace_dict.get("time_base_retrieval")
                t_critic = trace_dict.get("time_critic_evaluation")
                t_comp = trace_dict.get("time_completion_retrieval")
                t_ans = trace_dict.get("time_answer_generation")

            except Exception as e:
                logging.error(f"Failed on {qid}: {e}")
                generated_answer = f"ERROR: {e}"
                is_correct = False
                retrieved_context = f"ERROR: {e}"

            t_total = time.time() - t_start

            rec = {
                "Question_ID": qid,
                "Type": q_type,
                "Question": question_text,
                "Generated_Answer": generated_answer,
                "Ground_Truth_Answer": row["Ground_Truth_Answer"],
                "Is_Correct": is_correct,
                "Retrieved_Context": retrieved_context,
                "Time_Total_Item": round(t_total, 2),
                "Time_Base_Retrieval": round(t_base, 2) if t_base is not None else "",
                "Time_Critic_Evaluation": round(t_critic, 2) if t_critic is not None else "",
                "Time_Completion_Retrieval": round(t_comp, 2) if t_comp is not None else "",
                "Time_Answer_Generation": round(t_ans, 2) if t_ans is not None else ""
            }
            
            results.append(rec)

            if (idx + 1) % checkpoint_every == 0:
                pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")

        out_df = pd.DataFrame(results)
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("📊 BENCHMARK ACCURACY REPORT")
    print(f"{'='*60}")
    for qt in out_df["Type"].unique():
        sub_df = out_df[out_df["Type"] == qt]
        acc = sub_df["Is_Correct"].mean() * 100
        print(f"  Type {qt} ({len(sub_df)} Qs): {sub_df['Is_Correct'].sum()}/{len(sub_df)} = {acc:.1f}%")
    total_acc = out_df["Is_Correct"].mean() * 100
    print(f"  Overall Accuracy: {out_df['Is_Correct'].sum()}/{len(out_df)} = {total_acc:.1f}%")
    print(f"{'='*60}")
    print(f"✅ Report saved to: {output_csv}\n")


def run_inference_mode(csv_path: str, limit: int, use_rag: bool, use_completion: bool):
    if not os.path.exists(csv_path):
        logging.error(f"Input CSV not found: {csv_path}")
        return

    try:
        df_questions = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV {csv_path}: {e}")
        return

    if df_questions.empty:
        logging.error("The provided CSV is empty.")
        return

    if "Question_ID" not in df_questions.columns:
        df_questions["Question_ID"] = [f"Custom_Q_{i:04d}" for i in range(len(df_questions))]

    if limit > 0:
        df_questions = df_questions.head(limit)

    output_dir = os.path.join(PROJECT_ROOT, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    ablation_tag = ""
    if not use_rag:
        ablation_tag += "_NO_RAG"
    if not use_completion:
        ablation_tag += "_NO_COMPLETION"

    output_csv = os.path.join(output_dir, f"{base_name}{ablation_tag}_results.csv")

    results = []
    processed_ids = set()

    if os.path.exists(output_csv):
        print(f"\n🔄 [RESUME MODE] Found existing output CSV: {output_csv}")
        try:
            existing_df = pd.read_csv(output_csv)
            if "Question_ID" in existing_df.columns:
                processed_ids = set(existing_df["Question_ID"].astype(str).tolist())
                results = existing_df.to_dict(orient="records")
                print(f"   Already processed {len(processed_ids)} questions. Resuming...")
        except Exception as e:
            logging.warning(f"Could not load existing CSV for resume: {e}")

    remaining_df = df_questions[~df_questions["Question_ID"].astype(str).isin(processed_ids)].reset_index(drop=True)

    if remaining_df.empty:
        print(f"\n✅ All questions in {csv_path} have already been processed!")
        return

    print(f"\n🚀 [INFERENCE MODE] Evaluating {len(remaining_df)} remaining questions from CSV (Sequential & Safe)...")
    print(f"🔧 Ablation Settings: Use RAG = {use_rag}, Use Completion = {use_completion}")
    checkpoint_every = 5

    for idx, row in tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Running Inference"):
        qid = row["Question_ID"]
        question_text = row.get("Question", row.get("question", ""))
        q_type = row.get("Type", row.get("type", "QA")) 

        if not question_text or pd.isna(question_text):
            logging.warning(f"Row {qid} missing 'Question' text. Skipping.")
            continue

        t_start = time.time()
        retrieved_context = ""
        t_base, t_critic, t_comp, t_ans = None, None, None, None

        try:
            pipeline_output = run_pipeline(
                question=question_text, 
                q_type=q_type, 
                question_id=qid,
                use_rag=use_rag,
                use_completion=use_completion
            )
            generated_answer = pipeline_output.get("final_answer", "")
            retrieved_context = pipeline_output.get("final_context", "")
            trace_dict = pipeline_output.get("trace", {})

            t_base = trace_dict.get("time_base_retrieval")
            t_critic = trace_dict.get("time_critic_evaluation")
            t_comp = trace_dict.get("time_completion_retrieval")
            t_ans = trace_dict.get("time_answer_generation")

        except Exception as e:
            logging.error(f"Failed on {qid}: {e}")
            generated_answer = f"ERROR: {e}"
            retrieved_context = f"ERROR: {e}"

        t_total = time.time() - t_start

        rec = row.to_dict()
        rec.update({
            "Generated_Answer": generated_answer,
            "Retrieved_Context": retrieved_context,
            "Time_Total_Item": round(t_total, 2),
            "Time_Base_Retrieval": round(t_base, 2) if t_base is not None else "",
            "Time_Critic_Evaluation": round(t_critic, 2) if t_critic is not None else "",
            "Time_Completion_Retrieval": round(t_comp, 2) if t_comp is not None else "",
            "Time_Answer_Generation": round(t_ans, 2) if t_ans is not None else ""
        })
        
        results.append(rec)

        if (idx + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Inference complete. Report saved to: {output_csv}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Online RAG Pipeline")
    parser.add_argument(
        "--subset",
        choices=["mc", "tf", "all"],
        default="all",
        help="JSON dataset subset to run (Benchmark Mode)",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated numeric IDs from the JSON 'ID' field to run in Benchmark Mode.",
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to a custom CSV file (Inference Mode)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of questions to process")
    
    # Ablation Flags
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG completely (pure generation)")
    parser.add_argument("--no-completion", action="store_true", help="Disable Critic and automatic supplementary retrieval")

    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    missing_keys = []
    if not openai_key:
        missing_keys.append("OPENAI_API_KEY")
    if not google_key:
        missing_keys.append("GOOGLE_API_KEY")

    if missing_keys:
        print("\n" + "=" * 70)
        print("❌ [CRITICAL ERROR] Pre-flight Check Failed!")
        print("=" * 70)
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease set both environment variables before starting.")
        sys.exit(1)

    use_rag = not args.no_rag
    use_completion = not args.no_completion

    if args.csv:
        run_inference_mode(args.csv, args.limit, use_rag=use_rag, use_completion=use_completion)
    else:
        run_benchmark_mode(args.subset, args.limit, args.ids, use_rag=use_rag, use_completion=use_completion)


if __name__ == "__main__":
    main()