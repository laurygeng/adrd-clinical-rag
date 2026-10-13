import os
import time
import re
import json
import pandas as pd
from llm_engine import LocalLLMClient

# === 配置区 ===
# 指向你的 MC 错题表 (以 Lingshu 刚才那份为例，你可以随时换成 Meerkat 或 Hulu)
MC_CSV_PATH = 'benchmark_eval_mc_lingshu_medical_mllm_Lingshu_7B_20261012_234852.csv' 

model_id = os.environ.get("GLOBAL_AGENT_MODEL")
if not model_id:
    raise ValueError("❌ 未检测到模型！请先设置 GLOBAL_AGENT_MODEL")

client = LocalLLMClient()

print(f"🚀 开始运行 MC 题 [提纯策略 (Refinement)] 消融实验...")
print(f"🤖 当前测试模型: {model_id}\n")

# =====================================================================
# 1. 定义三种提纯策略 (Refinement Strategies)
# =====================================================================

def strategy_1_no_refinement(text, query):
    """策略 1：完全不提纯 (Raw Bypass) - 测试海量干扰项对 MC 的影响"""
    return str(text)[:1200]

def strategy_2_old_baseline(text, query):
    """策略 2：旧版经典提纯 - 存在信息过载和提取泛化内容的风险"""
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
    """策略 3：强力 JSON CoT - 严格过滤废话，专门针对 MC 设计"""
    prompt = (
        "Task: Output valid JSON ONLY with two exact keys:\n"
        "- \"reasoning\": 1 short sentence evaluating if the text contains the SPECIFIC answer/facts to the query, or if it lacks them / is just generic background advice.\n"
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
# 2. 答题引擎 (Answer Agent 模拟)
# =====================================================================
def get_final_answer(context, question):
    sys_prompt = "You are an expert medical exam taker. Output EXACTLY ONE uppercase letter (A, B, C, D, or E). Do not output anything else."
    user_prompt = f"Read this context carefully:\n{context}\n\nNow answer this question:\n{question}\n\nOutput ONLY the uppercase letter (A, B, C, D, or E)."
    try:
        r = client.chat.completions.create(model=model_id, temperature=0.0, max_tokens=10, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])
        ans = r.choices[0].message.content.strip().upper()
        # 提取单独的选项字母
        match = re.search(r'\b([A-E])\b', ans)
        return match.group(1) if match else ans
    except:
        return "ERROR"

# =====================================================================
# 3. 全量错题遍历执行
# =====================================================================
def run_refinement_ablation():
    if not os.path.exists(MC_CSV_PATH):
        print(f"⚠️ 找不到文件: {MC_CSV_PATH}")
        return

    df = pd.read_csv(MC_CSV_PATH)
    wrong_df = df[(df['Is_Correct'] == False) & (df['Retrieved_Passages'].notna())]
    print(f"📦 正在测试 MC 题库中的 {len(wrong_df)} 道错题...\n")

    results = {"No_Refine": 0, "Old_Baseline": 0, "JSON_CoT": 0}
    total = len(wrong_df)

    if total == 0:
        print("没有找到符合条件的错题！")
        return

    for _, row in wrong_df.iterrows():
        q_id = row['Question_ID']
        question = row['Question']
        gt = str(row['Ground_Truth_Answer']).strip().upper()
        raw_context = row['Retrieved_Passages'] 

        print(f"🎯 题目: [{q_id}] | 标准答案: {gt}")

        ctx_1 = strategy_1_no_refinement(raw_context, question)
        ctx_2 = strategy_2_old_baseline(raw_context, question)
        ctx_3 = strategy_3_json_cot(raw_context, question)

        ans_1 = get_final_answer(ctx_1, question)
        ans_2 = get_final_answer(ctx_2, question)
        ans_3 = get_final_answer(ctx_3, question)

        is_corr_1 = 1 if ans_1 == gt else 0
        is_corr_2 = 1 if ans_2 == gt else 0
        is_corr_3 = 1 if ans_3 == gt else 0

        results["No_Refine"] += is_corr_1
        results["Old_Baseline"] += is_corr_2
        results["JSON_CoT"] += is_corr_3

        print(f"  [1. 无提纯直出]   输出: {ans_1} {'✅' if is_corr_1 else '❌'}")
        print(f"  [2. 旧版经典提纯] 输出: {ans_2} {'✅' if is_corr_2 else '❌'}")
        print(f"  [3. 强力 JSON CoT] 输出: {ans_3} {'✅' if is_corr_3 else '❌'}\n")

    print(f"📊 提纯策略对 MC 错题的挽救率 (共 {total} 题):")
    print(f"  -> 1. 无提纯直出: {(results['No_Refine']/total)*100:.1f}%")
    print(f"  -> 2. 旧版经典提纯: {(results['Old_Baseline']/total)*100:.1f}%")
    print(f"  -> 3. 强力 JSON CoT: {(results['JSON_CoT']/total)*100:.1f}%")

if __name__ == "__main__":
    run_refinement_ablation()