#!/usr/bin/env python3
"""
Evaluate Gemini 3 Flash on ADRD-Bench (Multiple Choice + True/False).
Loads questions from local JSON files. Supports optional RAG context injection.

Includes retry logic with exponential backoff for 429 rate limits,
and enforced 12s sleep between requests (5 RPM limit).

Usage (from code/ directory):
  # No-RAG mode:
  python generate_answers/generate_answers_gemini3_flash_ADRD_Bench.py

  # With RAG retrieval context:
  python generate_answers/generate_answers_gemini3_flash_ADRD_Bench.py --retrieval ./retrieve_results/retrieval_ADRD_all_k3_w500_20260404_164315.csv
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import google.generativeai as genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_generation_config import SYSTEM_PROMPT, GEN_CONFIG, format_user_prompt, build_context_from_passages, MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# CONFIGURATION
# ==========================================
_cfg = MODEL_REGISTRY["gemini3_flash"]
MODEL_NAME = _cfg.model_id

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


# ==========================================
# DATA LOADING
# ==========================================

def load_adrd_bench(subset="all"):
    """Load ADRD-Bench questions from local JSON files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    mc_path = os.path.join(data_dir, "ADRD_Caregiving_Multiple_Choice.json")
    tf_path = os.path.join(data_dir, "ADRD_Caregiving_True_or_False.json")
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
    if df.empty:
        print(f"❌ No questions loaded from {data_dir}. Check the JSON file paths.")
        return df
    mc_n = len(df[df["Type"] == "MC"])
    tf_n = len(df[df["Type"] == "TF"])
    print(f"\n📊 Total: {len(df)} questions  ({mc_n} MC + {tf_n} T/F)\n")
    return df


# ==========================================
# GENERATION (Google Gemini 3 Flash)
# ==========================================

def generate_answer(model, question, context="", q_type="MC"):
    """Call Gemini 3 Flash with optional RAG context, with retry logic."""
    user_content = format_user_prompt(context, question)
    full_prompt = f"System Instructions: {SYSTEM_PROMPT}\n\nUser Input: {user_content}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=GEN_CONFIG.gemini_max_output_tokens,
                ),
                safety_settings=SAFETY_SETTINGS,
            )
            return response.text.strip() if response.text else "Error: Empty Response"
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                wait_time = 60 * (attempt + 1)
                logging.warning(f"⚠️ Rate limited (429). Retrying in {wait_time}s (attempt {attempt+1})...")
                time.sleep(wait_time)
            else:
                logging.error(f"Gemini API error: {error_msg}")
                return f"Error: {error_msg}"

    return "Error: Max retries exceeded"


# ==========================================
# ACCURACY CHECK
# ==========================================

def check_accuracy(generated, ground_truth, correct_letter, q_type):
    if not generated:
        return False
        
    text = generated.strip().upper()
    first_token = text.split()[0].rstrip(".,!?:") if text else ""
    
    if q_type == "TF":
        gt = str(ground_truth).strip().upper()
        yes_variants = ("YES", "TRUE", "T")
        no_variants = ("NO", "FALSE", "F")
        
        if gt in ("YES", "TRUE"):
            return first_token in yes_variants
        else:
            return first_token in no_variants
        
    elif q_type == "MC":
        return first_token == str(correct_letter).strip().upper()
        
    return False


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemini 3 Flash on ADRD-Bench")
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

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: Please set the GOOGLE_API_KEY environment variable.")
        return
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=MODEL_NAME)

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
            try:
                scores = json.loads(row["Rerank_Scores"])
            except Exception:
                scores = []
            # Preserve TF_Verdict from retrieval CSV so generation can honor evidence-first conclusion
            tf_verdict = None
            try:
                if "TF_Verdict" in row and pd.notna(row["TF_Verdict"]):
                    tf_verdict = str(row["TF_Verdict"]).strip()
            except Exception:
                tf_verdict = None
            retrieval_map[row["Question_ID"]] = {"passages": passages, "scores": scores, "tf_verdict": tf_verdict}
        print(f"✅ Loaded RAG context for {len(retrieval_map)} questions.\n")

    # Generation loop
    est_min = len(df) * 12 / 60
    print(f"🚀 Starting generation for {len(df)} questions (est. ~{est_min:.0f} min with 12s interval)...\n")
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Gemini3-Flash ADRD-Bench"):
        q_text = row["Question"]
        q_type = row["Type"]
        context = ""
        if retrieval_map:
            entry = retrieval_map.get(row["Question_ID"], {})
            if entry:
                context = build_context_from_passages(
                    entry.get("passages", []),
                    GEN_CONFIG.max_context_snippets,
                    scores=entry.get("scores", []),
                )
                # If TF and a TF_Verdict exists from the retrieval step, inject it as a high-priority hint
                if q_type == "TF" and entry.get("tf_verdict"):
                    # Map retrieval TF verdict (True/False) to dataset format (Yes/No)
                    verdict_str = entry.get('tf_verdict').strip()
                    mapped_verdict = "Yes" if verdict_str.lower() == "true" else "No"

                    verdict_hint = (
                        f"[System Fact-Check Module Conclusion]: Evidence heavily points to {mapped_verdict}."
                    )
                    context = context + "\n\n" + verdict_hint + "\n\n" + (
                        "Please answer with ONLY Yes or No based strictly on the Context and the Fact-Check Conclusion."
                    )

        generated = generate_answer(model, q_text, context, q_type=q_type)
        is_correct = check_accuracy(generated, row["Ground_Truth_Answer"], row["Correct_Letter"], row["Type"])

        results.append({
            "Question_ID": row["Question_ID"], "Type": row["Type"],
            "Question": q_text, "Generated_Answer": generated,
            "Ground_Truth_Answer": row["Ground_Truth_Answer"],
            "Correct_Letter": row["Correct_Letter"], "Is_Correct": is_correct,
        })

        # Enforce 12s interval to respect Gemini API rate limit (5 RPM)
        time.sleep(12)

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
