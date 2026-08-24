#!/usr/bin/env python3
import os
import sys
import json
import gc
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 把 core 目录加到系统路径，方便导包
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core"))

from core.answer_agent import check_accuracy

def main():
    # 检测 GPU 存活状态 (防止在 CPU 上龟速运行)
    if not torch.cuda.is_available():
        print("\n" + "!"*60)
        print("🚨 [硬件危机] PyTorch 无法连接到 GPU！")
        print("系统试图在 CPU 上运行大模型，这会导致耗时极长且答案全错。")
        print("请检查你的 CUDA 环境，或重新 load cuda 模块，修复后再运行此脚本！")
        print("!"*60 + "\n")
        sys.exit(1)

    input_csv = os.path.join(PROJECT_ROOT, "evaluation_results", "benchmark_eval_mc_20260822_232757.csv")
    output_csv = os.path.join(PROJECT_ROOT, "evaluation_results", "benchmark_eval_mc_REGENERATED.csv")
    json_path = os.path.join(PROJECT_ROOT, "data", "ADRD_Caregiving_Multiple_Choice.json")

    if not os.path.exists(input_csv):
        print(f"❌ 找不到文件: {input_csv}")
        return

    print("📖 正在从原始 JSON 找回正确选项字母...")
    gt_map = {}
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            for item in json.load(f).get("data", []):
                gt_map[f"ADRD_MC_{int(item.get('ID')):03d}"] = item.get("Answer", "")

    print(f"✅ 加载数据文件: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print("🤖 正在启动 Hulu-Med-7B 推理服务...")
    model_id = os.environ.get("LOCAL_AGENT_A_MODEL", "ZJU-AI4H/Hulu-Med-7B")
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True
    )

    print("\n🚀 开始纯生成模式 (Generation Only)...")
    results = []
    
    system_prompt = (
        "You are an expert medical AI assistant specializing in Alzheimer's Disease and Related Dementias (ADRD). "
        "Your core directive is to answer the user's question STRICTLY based on the provided retrieved context. "
        "Do NOT use external knowledge. If the context does not contain the answer, do not guess."
    )

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Answers"):
        qid = row["Question_ID"]
        q_type = row["Type"]
        question_text = row["Question"]
        context = str(row["Retrieved_Context"])
        ground_truth = row["Ground_Truth_Answer"]
        correct_letter = gt_map.get(qid, "")

        user_content = (
            f"--- Retrieved Context ---\n{context}\n\n"
            f"--- Question ---\n{question_text}\n\n"
            "--- INSTRUCTIONS ---\n"
            "1. GROUNDING: Answer STRICTLY based on the provided context. Do NOT use outside knowledge.\n"
            "2. FORMAT: Output ONLY the final answer, with no explanation, preamble, or extra text.\n"
            "3. Your output must be EXACTLY ONE option letter (A, B, C, D, or E) — the one best supported by the context."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_prompt = f"System: {system_prompt}\nUser: {user_content}\nAssistant:\n"

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        generated_answer = "FAILED_EMPTY_OUTPUT"
        try:
            with torch.no_grad():
                # 修复1：强制使用 kwargs 关键字传参，绝不能用位置传参
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=32,
                    do_sample=False
                )
            
            # 修复2：智能防越界切片
            seq = outputs[0]
            input_length = inputs.input_ids.shape[1]
            
            # 判断模型是否把 prompt 吐出来了，如果吐出来了就截断，如果只有新 token 就直接取用
            if len(seq) > input_length:
                generated_tokens = seq[input_length:].tolist()
            else:
                generated_tokens = seq.tolist()
                
            clean_decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            if clean_decoded:
                generated_answer = clean_decoded
                
        except Exception as e:
            print(f"\n❌ Error on {qid}: {e}")
        finally:
            del inputs
            if 'outputs' in locals(): del outputs
            torch.cuda.empty_cache()
            gc.collect()

        is_correct = check_accuracy(generated_answer, ground_truth, correct_letter, q_type)

        rec = row.to_dict()
        rec["Generated_Answer"] = generated_answer
        rec["Is_Correct"] = is_correct
        rec["Correct_Letter"] = correct_letter 
        results.append(rec)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("📊 重新生成成绩单 (REGENERATED REPORT)")
    print(f"{'='*60}")
    acc = out_df["Is_Correct"].mean() * 100
    print(f"  Overall Accuracy: {out_df['Is_Correct'].sum()}/{len(out_df)} = {acc:.1f}%")
    print(f"{'='*60}")
    print(f"✅ 修复后的报告已保存至: {output_csv}\n")

if __name__ == "__main__":
    main()