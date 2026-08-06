#!/usr/bin/env python3
"""
Refined ADRD-Bench Evaluator with LLM-based Context Filtering.
This script now filters passages using 'extract_and_cite_facts' before generating.
"""

import os
import json
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# 导入你的核心过滤工具
from llm_utils import extract_and_cite_facts, get_openai_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_passages(question, passages):
    """
    【预处理层：文档打分与提炼】
    利用 llm_utils.py 的 extract_and_cite_facts 提取事实。
    如果一段文档无法被提取出事实，说明它对回答问题无贡献，直接过滤掉。
    """
    try:
        # 传入原始 passages，强制模型只保留包含事实的文本
        facts_json = extract_and_cite_facts(question, passages)
        data = json.loads(facts_json)
        facts = data.get("extracted_facts", [])
        
        # 仅保留被引用过的原始段落，剔除无效垃圾信息
        relevant_indices = set(item["source_index"] for item in facts if "source_index" in item)
        refined = [passages[i] for i in range(len(passages)) if i in relevant_indices]
        
        return refined if refined else passages # 如果提取失败，降级保留原始段落
    except Exception as e:
        logging.warning(f"Filtering failed, keeping original: {e}")
        return passages

# def generate_answer_refined(client, question, passages, model_name):
# def generate_answer_refined(client, question, passages, qtype, model_name):
#     """
#     【生成层：聚焦生成】
#     只接收经过 filter_passages 提纯后的上下文。
#     """
#     refined_context = filter_passages(question, passages)
#     context_str = "\n\n".join(refined_context)
    
#     # 构造聚焦式 Prompt
#     prompt = f"Context: {context_str}\n\nQuestion: {question}\n\nAnswer concisely based ONLY on the context above."
    
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.1
#     )
#     return response.choices[0].message.content.strip()
def generate_answer_refined(client, question, passages, qtype, model_name):
    """
    【升级版生成层】：注入严格的 SYSTEM_PROMPT 约束
    """
    # 1. 过滤噪音
    refined_context = filter_passages(question, passages)
    context_str = "\n\n".join(refined_context)
    
    # 2. 定义你要求的严格 SYSTEM_PROMPT
    system_prompt = (
        "You are a supportive expert assistant for dementia caregivers. "
        "You will be given relevant medical passages retrieved from authoritative sources, followed by a question. "
        "Use ONLY the provided context to answer. If the information is not in the context, "
        "state that you do not know. DO NOT use outside knowledge.\n\n"
        "IMPORTANT FORMATTING RULES:\n"
        "- For True/False questions: You MUST start your response with exactly 'Yes' or 'No' (the very first word), "
        "then provide a brief explanation.\n"
        "- For Multiple Choice questions: You MUST start your response with the letter of the correct answer "
        "followed by a period (e.g., 'A. ...', 'B. ...'), then provide the answer text and explanation."
    )
    
    # 3. 构造聚焦 Prompt，防止复述问题
    # 这里我们明确告诉模型不要重复问题，直接给出答案
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer concisely according to the rules:"
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # 温度调为0，最大限度增强确定性和遵循指令的能力
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def check_accuracy(generated, ground_truth, correct_letter, q_type):
    """
    Utility to verify if the generated answer matches the ground truth.
    """
    if not generated:
        return False
    
    # Clean the generated answer to get the first token
    first_token = str(generated).strip().split()[0].rstrip(".,!?:").upper()
    
    if q_type == "TF":
        return first_token == str(ground_truth).strip().upper()
    elif q_type == "MC":
        if correct_letter:
            return first_token == str(correct_letter).strip().upper()
        return str(ground_truth).strip().lower() in str(generated).strip().lower()
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str, required=True, help="Path to retrieval CSV")
    args = parser.parse_args()

    client = get_openai_client()
    df = pd.read_csv(args.retrieval)
    results = []

    for _, row in tqdm(df.iterrows(), desc="Generating with Refinement"):
        passages = json.loads(row["Retrieved_Passages"])
        
        # 使用提纯后的生成
        # answer = generate_answer_with_refinement(client, row["Question"], passages, "gpt-4o")
        # answer = generate_answer_refined(client, row["Question"], passages, row["Type"], "gpt-4o")
        # 必须是 5 个参数，一一对应
        answer = generate_answer_refined(client, row["Question"], passages, row["Type"], "gpt-4o")
        results.append({
            "Question_ID": row["Question_ID"],
            "Generated_Answer": answer,
            "Is_Correct": check_accuracy(answer, row["Ground_Truth_Answer"], row.get("Correct_Letter"), row["Type"])
        })
        
    pd.DataFrame(results).to_csv("final_refined_results.csv")

if __name__ == "__main__":
    main()