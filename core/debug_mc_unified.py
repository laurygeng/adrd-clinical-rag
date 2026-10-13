import os
import re
import json
import pandas as pd
from llm_engine import LocalLLMClient

# === 配置区 ===
MC_CSV_PATH = 'benchmark_eval_mc_lingshu_medical_mllm_Lingshu_7B_20261012_234852.csv' 

model_id = os.environ.get("GLOBAL_AGENT_MODEL")
if not model_id:
    raise ValueError("❌ 未检测到模型！请先设置 GLOBAL_AGENT_MODEL")

client = LocalLLMClient()

print(f"🚀 开始运行 MC 终极提纯消融实验 (7道错题 + 10道对题)...")
print(f"🤖 当前测试模型: {model_id}\n")

# =====================================================================
# 1. 定义三种提纯策略
# =====================================================================
def strategy_1_no_refinement(text, query):
    return str(text)[:1200]

def strategy_2_old_baseline(text, query):
    prompt = (
        f"Extract the specific paragraphs and clinical facts from the text below that directly answer or verify: '{query}'. "
        "Discard unrelated chit-chat or website navigation, but KEEP the full context of the relevant medical facts. Do NOT over-compress.\n\n"
        f"Text:\n{str(text)[:1200]}\n\nExtracted Facts:"
    )
    try:
        r = client.chat.completions.create(model=model_id, temperature=0.0, max_tokens=250, messages=[{"role": "user", "content": prompt}])
        ans = r.choices[0].message.content.strip()
        return ans if ans and not _looks_unknown(ans) else str(text)[:400]
    except:
        return str(text)[:400]

def strategy_3_json_cot(text, query):
    prompt = (
        "Task: Output valid JSON ONLY with two exact keys:\n"
        "- \"reasoning\": 1 short sentence evaluating if the text contains the SPECIFIC answer/facts to the query.\n"
        "- \"extracted\": The specific clinical facts, or \"NONE\" if the text lacks the specific answer.\n"
        "CRITICAL FOR MULTIPLE CHOICE: Do NOT extract broad generic advice. If the text only has generic advice but no specific answer, 'extracted' MUST be 'NONE'.\n\n"
        f"Query: '{query}'\nText:\n{str(text)[:1200]}\n\nJSON:"
    )
    try:
        r = client.chat.completions.create(model=model_id, temperature=0.0, max_tokens=250, messages=[{"role": "user", "content": prompt}])
        raw = r.choices[0].message.content.strip()
        raw = re.split(r"(?i)### response", raw)[0].strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            ext = str(data.get("extracted", "NONE")).strip()
            return "" if ext.upper() == "NONE" else ext
        return ""
    except:
        return ""

def _looks_unknown(text):
    return any(p in text.lower() for p in ["don't know", "not enough", "insufficient"])

# =====================================================================
# 2. 答题引擎与判分修复
# =====================================================================
def get_final_answer(context, question):
    sys_prompt = "You are an expert medical exam taker. Output EXACTLY ONE uppercase letter (A, B, C, D, or E). Do not output anything else."
    user_prompt = f"Read this context carefully:\n{context}\n\nNow answer this question:\n{question}\n\nOutput ONLY the uppercase letter (A, B, C, D, or E)."
    try:
        r = client.chat.completions.create(model=model_id, temperature=0.0, max_tokens=10, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])
        ans = r.choices[0].message.content.strip().upper()
        match = re.search(r'\b([A-E])\b', ans)
        return match.group(1) if match else ans
    except:
        return "ERROR"

def _get_gt_letter(question, gt_text):
    """从题目文本中寻找标准答案文本对应的字母选项"""
    gt_clean = str(gt_text).strip().lower()
    for line in question.split('\n'):
        line = line.strip()
        if re.match(r'^[A-E][\.\)]', line) and gt_clean in line.lower():
            return line[0].upper()
    return gt_clean 

# =====================================================================
# 3. 提取 7错 + 10对 进行验证
# =====================================================================
def run_refinement_ablation():
    if not os.path.exists(MC_CSV_PATH):
        print(f"⚠️ 找不到文件: {MC_CSV_PATH}")
        return

    df = pd.read_csv(MC_CSV_PATH)
    
    wrong_df = df[(df['Is_Correct'] == False) & (df['Retrieved_Passages'].notna())].copy()
    wrong_df['Original_Status'] = 'WRONG'
    
    correct_df = df[(df['Is_Correct'] == True) & (df['Retrieved_Passages'].notna())].copy()
    sample_size = min(10, len(correct_df))
    correct_sample = correct_df.sample(n=sample_size, random_state=42)
    correct_sample['Original_Status'] = 'CORRECT'
    
    test_df = pd.concat([wrong_df, correct_sample])
    print(f"📦 共组装了 {len(test_df)} 道题 (错题: {len(wrong_df)}, 对题: {len(correct_sample)})\n")

    results_wrong = {"No_Refine": 0, "Old_Baseline": 0, "JSON_CoT": 0}
    results_correct = {"No_Refine": 0, "Old_Baseline": 0, "JSON_CoT": 0}

    for _, row in test_df.iterrows():
        q_id = row['Question_ID']
        question = row['Question']
        gt_text = str(row['Ground_Truth_Answer']).strip().upper()
        gt_letter = _get_gt_letter(question, gt_text)
        raw_context = row['Retrieved_Passages'] 
        status = row['Original_Status']

        print(f"\n🎯 [{status}] 题目: {q_id} | 标答: {gt_letter} ({gt_text[:15]}...)")

        ctx_1 = strategy_1_no_refinement(raw_context, question)
        ctx_2 = strategy_2_old_baseline(raw_context, question)
        ctx_3 = strategy_3_json_cot(raw_context, question)

        ans_1 = get_final_answer(ctx_1, question)
        ans_2 = get_final_answer(ctx_2, question)
        ans_3 = get_final_answer(ctx_3, question)

        is_corr_1 = 1 if ans_1 == gt_letter else 0
        is_corr_2 = 1 if ans_2 == gt_letter else 0
        is_corr_3 = 1 if ans_3 == gt_letter else 0

        if status == 'WRONG':
            results_wrong["No_Refine"] += is_corr_1
            results_wrong["Old_Baseline"] += is_corr_2
            results_wrong["JSON_CoT"] += is_corr_3
        else:
            results_correct["No_Refine"] += is_corr_1
            results_correct["Old_Baseline"] += is_corr_2
            results_correct["JSON_CoT"] += is_corr_3

        print(f"  [1. 无提纯直出]   输出: {ans_1} {'✅' if is_corr_1 else '❌'}")
        print(f"  [2. 旧版经典提纯] 输出: {ans_2} {'✅' if is_corr_2 else '❌'}")
        print(f"  [3. 强力 JSON CoT] 输出: {ans_3} {'✅' if is_corr_3 else '❌'}")

    # --- 打印统计结果 ---
    print("\n" + "="*50)
    print("📊 终极消融实验统计结果")
    print("="*50)
    
    if len(wrong_df) > 0:
        print(f"📌 错题挽救率 (共 {len(wrong_df)} 题 - 越高越好):")
        print(f"  -> 1. 无提纯直出: {(results_wrong['No_Refine']/len(wrong_df))*100:.1f}%")
        print(f"  -> 2. 旧版经典提纯: {(results_wrong['Old_Baseline']/len(wrong_df))*100:.1f}%")
        print(f"  -> 3. 强力 JSON CoT: {(results_wrong['JSON_CoT']/len(wrong_df))*100:.1f}%")

    if len(correct_sample) > 0:
        print(f"\n📌 正确题保持率 (共 {len(correct_sample)} 题 - 越高越好，防止副作用):")
        print(f"  -> 1. 无提纯直出: {(results_correct['No_Refine']/len(correct_sample))*100:.1f}%")
        print(f"  -> 2. 旧版经典提纯: {(results_correct['Old_Baseline']/len(correct_sample))*100:.1f}%")
        print(f"  -> 3. 强力 JSON CoT: {(results_correct['JSON_CoT']/len(correct_sample))*100:.1f}%")

if __name__ == "__main__":
    run_refinement_ablation()