#!/usr/bin/env python3
"""
Online Pipeline Batch Runner (entrypoints/)
Role:
1) Benchmark Mode (JSONs): Evaluate ADRD MC/TF, calculate accuracy, export CSV (with full Retrieved_Context and traces).
2) Inference Mode (CSV): Process custom CSV, append answers + contexts + traces.
"""

import os
import sys
import json
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

            records.append(
                {
                    "Question_ID": q_id,
                    "Type": "MC",
                    "Question": formatted_q,
                    "Ground_Truth_Answer": ground_truth_text,
                    "Correct_Letter": ans_letter,
                    "Numeric_ID": q_num,
                }
            )

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

            records.append(
                {
                    "Question_ID": q_id,
                    "Type": "TF",
                    "Question": formatted_q,
                    "Ground_Truth_Answer": ground_truth,
                    "Correct_Letter": ground_truth,
                    "Numeric_ID": q_num,
                }
            )

    df = pd.DataFrame(records)
    if not df.empty and "Numeric_ID" in df.columns:
        df = df.sort_values(["Type", "Numeric_ID"], ascending=[True, True]).reset_index(drop=True)
    return df


def _extract_trace_fields(pipeline_output: dict) -> Dict[str, Any]:
    trace = (pipeline_output or {}).get("trace", {}) if isinstance(pipeline_output, dict) else {}
    out: Dict[str, Any] = {}

    out["Retrieved_Context"] = (pipeline_output or {}).get("final_context", "")
    out["Completion_Triggered"] = trace.get("completion_triggered", False)
    out["Completion_Reason"] = trace.get("completion_reason", "")
    out["Completion_Skipped_Reason"] = trace.get("completion_skipped_reason", "")
    out["Web_Query_Used_Hop1"] = trace.get("web_query_used", "")
    out["Web_Query_Used_Hop2"] = trace.get("web_query_used_hop2", "")
    out["Hop2_Triggered"] = trace.get("hop2_triggered", False)
    out["Hop2_Reason"] = trace.get("hop2_reason", "")
    out["Court_Statement_Used"] = trace.get("court_statement_used", "")
    out["Court_Verdict"] = trace.get("court_verdict", "N/A")
    out["Veto_Triggered"] = trace.get("veto_triggered", False)

    out["Critic_Is_Sufficient"] = trace.get("is_sufficient")
    out["Critic_Missing_Info"] = trace.get("missing_info", "")
    out["Critic_Decision"] = trace.get("critic_decision", "")
    out["Critic_None_Frac_Strict"] = trace.get("critic_none_frac_strict")
    out["Critic_Empty_Frac"] = trace.get("critic_empty_frac")
    out["Critic_Invalid_Gap_Frac"] = trace.get("critic_invalid_gap_frac")
    out["Critic_Consensus_Gap"] = trace.get("critic_consensus_gap", "")
    out["Critic_Consensus_Gap_Is_Negative"] = trace.get("critic_consensus_gap_is_negative", None)
    out["Critic_Verify_Mode"] = trace.get("critic_verify_mode", "")
    out["Critic_Verify_Label"] = trace.get("critic_verify_label", "")
    out["Critic_Verify_Best_Score"] = trace.get("critic_verify_best_score", None)
    out["Critic_Verify_Threshold"] = trace.get("critic_verify_threshold", None)
    out["Critic_Verify_Best_Span"] = trace.get("critic_verify_best_span", "")
    out["Critic_Verify_N_Spans"] = trace.get("critic_verify_n_spans", None)
    out["Critic_Verify_Window_Sents"] = trace.get("critic_verify_window_sents", None)
    out["Critic_MD_Path"] = trace.get("critic_md_path", "")

    return out


def run_benchmark_mode(subset: str, limit: int, ids_str: Optional[str]):
    allowed_ids = _parse_ids(ids_str)
    df_questions = load_json_benchmarks(subset=subset, allowed_ids=allowed_ids)
    if df_questions.empty:
        logging.error("No JSON benchmark questions found (or none matched --ids).")
        return

    if limit > 0:
        df_questions = df_questions.head(limit)

    output_dir = os.path.join(PROJECT_ROOT, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ids_tag = ""
    if allowed_ids is not None:
        ids_sorted = ",".join(str(x) for x in sorted(allowed_ids))
        ids_tag = f"_ids_{ids_sorted.replace(',', '-')}"
    output_csv = os.path.join(output_dir, f"benchmark_eval_{subset}_{timestamp}{ids_tag}.csv")

    print(f"\n🚀 [BENCHMARK MODE] Evaluating {len(df_questions)} questions...")

    results = []
    checkpoint_every = 10

    for idx, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="Running Benchmarks"):
        qid = row["Question_ID"]
        q_type = row["Type"]
        question_text = row["Question"]

        try:
            pipeline_output = run_pipeline(question=question_text, q_type=q_type, question_id=qid)
            generated_answer = pipeline_output.get("final_answer", "")

            is_correct = check_accuracy(
                generated=generated_answer,
                ground_truth=row["Ground_Truth_Answer"],
                correct_letter=row["Correct_Letter"],
                q_type=q_type,
            )

            trace_fields = _extract_trace_fields(pipeline_output)

        except Exception as e:
            logging.error(f"Failed on {qid}: {e}")
            generated_answer = f"ERROR: {e}"
            is_correct = False
            trace_fields = {"Retrieved_Context": ""}

        rec = {
            "Question_ID": qid,
            "Type": q_type,
            "Question": question_text,
            "Generated_Answer": generated_answer,
            "Ground_Truth_Answer": row["Ground_Truth_Answer"],
            "Is_Correct": is_correct,
        }
        rec.update(trace_fields)
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

    run_benchmark_mode(args.subset, args.limit, args.ids)


if __name__ == "__main__":
    main()