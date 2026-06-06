#!/usr/bin/env python3
"""
Evaluate GPT-5.2 on ADRD-Bench (Multiple Choice + True/False).
Loads questions from local JSON files. Supports optional RAG context injection.

Note: GPT-5.2 uses max_completion_tokens instead of max_tokens,
      and does not support custom temperature (uses default).

Usage (from code/ directory):
  # No-RAG mode:
  python generate_answers/generate_answers_gpt5.2_ADRD_Bench.py

  # With RAG retrieval context:
  python generate_answers/generate_answers_gpt5.2_ADRD_Bench.py --retrieval ./retrieve_results/retrieval_ADRD_all_k3_w500_20260404_164315.csv
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_generation_config import SYSTEM_PROMPT, GEN_CONFIG, format_user_prompt, build_context_from_passages, MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# CONFIGURATION
# ==========================================
_cfg = MODEL_REGISTRY["gpt5.2"]
MODEL_NAME = _cfg.model_id


# ==========================================
# DATA LOADING
# ==========================================

def load_adrd_bench(subset="all"):
    """Load ADRD-Bench questions from local JSON files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mc_path = os.path.join(script_dir, "ADRD_Caregiving_Multiple_Choice.json")
    tf_path = os.path.join(script_dir, "ADRD_Caregiving_True_or_False.json")
    records = []

    if subset in ("mc", "all"):
        if not os.path.exists(mc_path):
            print(f"⚠️  MC file not found: {mc_path}")
        else:
            print(f"📥 Loading MC questions from {os.path.basename(mc_path)}...")
            with open(mc_path, encoding="utf-8") as f:
                mc_data = json.load(f)
            for item in mc_data["data"]:
                q_id = f"ADRD_MC_{item['ID']:03d}"
                q_text = item["Question"]
                options = item.get("Options", {})
                ans_letter = item["Answer"]
                ground_truth_text = options.get(ans_letter, ans_letter)
                options_str = "\n".join([f"  {k}. {v}" for k, v in options.items()])
                formatted_q = (
                    f"{q_text}\n\nOptions:\n{options_str}\n\n"
                    f"Answer with ONLY the correct letter (A/B/C/D/E)."
                )
                records.append({
                    "Question_ID": q_id, "Question": formatted_q,
                    "Ground_Truth_Answer": ground_truth_text,
                    "Correct_Letter": ans_letter, "Type": "MC",
                })
            print(f"  ✅ Loaded {sum(1 for r in records if r['Type']=='MC')} MC questions.")

    if subset in ("tf", "all"):
        if not os.path.exists(tf_path):
            print(f"⚠️  T/F file not found: {tf_path}")
        else:
            print(f"📥 Loading T/F questions from {os.path.basename(tf_path)}...")
            with open(tf_path, encoding="utf-8") as f:
                tf_data = json.load(f)
            before = len(records)
            for item in tf_data["data"]:
                q_id = f"ADRD_TF_{item['ID']:03d}"
                q_text = item["Question"]
                ground_truth = item["Answer"]
                formatted_q = f"True or False statement: {q_text}\n\nAnswer with ONLY Yes or No."
                records.append({
                    "Question_ID": q_id, "Question": formatted_q,
                    "Ground_Truth_Answer": ground_truth,
                    "Correct_Letter": ground_truth, "Type": "TF",
                })
            print(f"  ✅ Loaded {len(records) - before} T/F questions.")

    df = pd.DataFrame(records)
    mc_n = len(df[df["Type"] == "MC"])
    tf_n = len(df[df["Type"] == "TF"])
    print(f"\n📊 Total: {len(df)} questions  ({mc_n} MC + {tf_n} T/F)\n")
    return df


# ==========================================
# GENERATION (OpenAI GPT-5.2)
# ==========================================

def generate_answer(client, question, context=""):
    """Call GPT-5.2 via OpenAI API with optional RAG context."""
    user_content = format_user_prompt(context, question)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            # GPT-5.2 only supports default temperature
            # Use max_completion_tokens for GPT-5 series (not max_tokens)
            max_completion_tokens=GEN_CONFIG.openai_max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"API error: {e}")
        return f"Error: {e}"


# ==========================================
# ACCURACY CHECK
# ==========================================

def check_accuracy(generated, ground_truth, correct_letter, q_type):
    first_token = generated.strip().split()[0].rstrip(".,!?:").upper() if generated.strip() else ""
    if q_type == "TF":
        return first_token == ground_truth.strip().upper()
    elif q_type == "MC":
        return first_token == correct_letter.strip().upper()
    return False


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-5.2 on ADRD-Bench")
    parser.add_argument("--retrieval", type=str, default=None,
                        help="Path to retrieval results CSV (optional).")
    parser.add_argument("--subset", choices=["mc", "tf", "all"], default="all",
                        help="Which subset to evaluate: mc | tf | all")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    answers_dir = os.path.join(script_dir, "answers")
    os.makedirs(answers_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_label = "rag" if args.retrieval else "norag"
    output_csv = os.path.join(answers_dir, f"{_cfg.output_prefix}_ADRD_{args.subset}_{mode_label}_{timestamp}.csv")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Please set the OPENAI_API_KEY environment variable.")
        return
    client = OpenAI(api_key=api_key)

    print(f"🤖 Model : {MODEL_NAME}")
    print(f"📂 Output: {output_csv}")
    print(f"🔧 Mode  : {mode_label.upper()}")

    df = load_adrd_bench(subset=args.subset)
    if df.empty:
        print("❌ No questions loaded. Exiting.")
        return

    # Load retrieval context map (optional)
    retrieval_map = {}
    if args.retrieval:
        if not os.path.exists(args.retrieval):
            print(f"❌ Retrieval file not found: {args.retrieval}")
            return
        ret_df = pd.read_csv(args.retrieval)
        for _, row in ret_df.iterrows():
            try:
                passages = json.loads(row["Retrieved_Passages"])
            except Exception:
                passages = [str(row.get("Retrieved_Passages", ""))]
            retrieval_map[row["Question_ID"]] = passages
        print(f"✅ Loaded RAG context for {len(retrieval_map)} questions.\n")

    # Generation loop
    print(f"🚀 Starting generation for {len(df)} questions...\n")
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="GPT-5.2 ADRD-Bench"):
        q_text = row["Question"]
        context = ""
        if retrieval_map:
            passages = retrieval_map.get(row["Question_ID"], [])
            if passages:
                context = build_context_from_passages(passages, GEN_CONFIG.max_context_snippets)

        generated = generate_answer(client, q_text, context)
        is_correct = check_accuracy(generated, row["Ground_Truth_Answer"], row["Correct_Letter"], row["Type"])

        results.append({
            "Question_ID": row["Question_ID"], "Type": row["Type"],
            "Question": q_text, "Generated_Answer": generated,
            "Ground_Truth_Answer": row["Ground_Truth_Answer"],
            "Correct_Letter": row["Correct_Letter"], "Is_Correct": is_correct,
        })
        time.sleep(0.3)

    # Save
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Accuracy summary
    print(f"\n{'='*55}")
    print(f"📊 Accuracy Summary  (model: {MODEL_NAME})")
    print(f"{'='*55}")
    for q_type in ["MC", "TF"]:
        sub = out_df[out_df["Type"] == q_type]
        if sub.empty:
            continue
        acc = sub["Is_Correct"].mean() * 100
        print(f"  {q_type} ({len(sub):>3} Qs):  {sub['Is_Correct'].sum():>3}/{len(sub):>3}  = {acc:5.1f}%")
    total_acc = out_df["Is_Correct"].mean() * 100
    print(f"  {'Overall':12s}  {out_df['Is_Correct'].sum():>3}/{len(out_df):>3}  = {total_acc:5.1f}%")
    print(f"{'='*55}")
    print(f"\n✅ Results saved to:\n   {output_csv}\n")


if __name__ == "__main__":
    main()
