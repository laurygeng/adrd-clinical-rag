#!/usr/bin/env python3
"""
GPT-5.2 runner without TF_Verdict injection. Creates the same prompt as
the original `generate_answers_gpt5.2_ADRD_Bench.py` (context + question),
but otherwise matches the modified runner's safe defaults.
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

_cfg = MODEL_REGISTRY["gpt5.2"]
MODEL_NAME = _cfg.model_id


def load_adrd_bench(subset="all"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    mc_path = os.path.join(data_dir, "ADRD_Caregiving_Multiple_Choice.json")
    tf_path = os.path.join(data_dir, "ADRD_Caregiving_True_or_False.json")
    records = []
    if subset in ("mc", "all"):
        if os.path.exists(mc_path):
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
    if subset in ("tf", "all"):
        if os.path.exists(tf_path):
            with open(tf_path, encoding="utf-8") as f:
                tf_data = json.load(f)
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
    df = pd.DataFrame(records)
    return df


def generate_answer(client, question, context=""):
    user_content = format_user_prompt(context, question)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_completion_tokens=GEN_CONFIG.openai_max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"API error: {e}")
        return f"Error: {e}"


def check_accuracy(generated, ground_truth, correct_letter, q_type):
    first_token = generated.strip().split()[0].rstrip(".,!?:").upper() if generated.strip() else ""
    if q_type == "TF":
        gt = ground_truth.strip().upper()
        if gt == "YES" and first_token in ("YES", "TRUE"):
            return True
        if gt == "NO" and first_token in ("NO", "FALSE"):
            return True
        return first_token == gt
    elif q_type == "MC":
        return first_token == correct_letter.strip().upper()
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str, default=None)
    parser.add_argument("--subset", choices=["mc","tf","all"], default="all")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Please set the OPENAI_API_KEY environment variable.")
        return
    client = OpenAI(api_key=api_key)

    df = load_adrd_bench(subset=args.subset)
    if df.empty:
        print("No questions loaded.")
        return

    retrieval_map = {}
    if args.retrieval:
        if not os.path.exists(args.retrieval):
            print(f"Retrieval file not found: {args.retrieval}")
            return
        ret_df = pd.read_csv(args.retrieval)
        for _, row in ret_df.iterrows():
            try:
                passages = json.loads(row.get("Retrieved_Passages","[]"))
            except Exception:
                passages = [str(row.get("Retrieved_Passages",""))]
            retrieval_map[row["Question_ID"]] = {"passages": passages}

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q_text = row["Question"]
        context = ""
        if retrieval_map:
            entry = retrieval_map.get(row["Question_ID"], {})
            passages = entry.get("passages", [])
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
        time.sleep(args.delay)

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "answers", f"{_cfg.output_prefix}_ADRD_{args.subset}_rag_no_verdict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
