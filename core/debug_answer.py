import os
import time
import ast
import pandas as pd
from llm_engine import LocalLLMClient

# 1. 从 CSV 动态读取真实的、充满噪音的长篇 Context
csv_path = '../benchmark_eval_mc_lingshu_medical_mllm_Lingshu_7B_20261011_025859.csv'
if not os.path.exists(csv_path):
    csv_path = 'benchmark_eval_mc_lingshu_medical_mllm_Lingshu_7B_20261011_025859.csv'

df = pd.read_csv(csv_path)
row = df[df['Question_ID'] == 'ADRD_MC_013'].iloc[0]

QUESTION = row['Question']
try:
    passages = ast.literal_eval(row['Retrieved_Passages'])
except:
    passages = [row['Retrieved_Passages']]
    
CONTEXT = "\n\n".join([str(p) for p in passages])

print(f"🚀 开始测试 Lingshu_7B 最终答题 (真实长文本压力测试)...\n")
print(f"📄 真实上下文长度: {len(CONTEXT)} 字符")
print(f"🎯 目标正确答案: C\n")

client = LocalLLMClient()
model_id = os.environ.get("GLOBAL_AGENT_MODEL", "ZJU-AI4H/Lingshu_7B")

def run_answer_experiment(name, sys_prompt, user_prompt, temp=0.1, max_tokens=20, prefill=""):
    print(f"{'='*60}")
    print(f"🧪 方案: {name}")
    try:
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
            
        t0 = time.time()
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens
        )
        text = response.choices[0].message.content.strip()
        
        is_correct = "C" in text or "c" in text
        status = "✅ 答对 (选了C)" if is_correct else "❌ 答错"
        
        print(f"⏱️ 耗时: {time.time()-t0:.2f}s | 判定: {status}")
        print(f"📤 原始输出: {text}")
    except Exception as e:
        print(f"❌ 运行失败: {e}")

# -------------------------------------------------------------------------
# 10 种答题防呆变异方案 (真实压力版)
# -------------------------------------------------------------------------

# 方案 1: Baseline (常规提问)
sys_1 = "You are a medical assistant. Answer the multiple choice question based on the context."
user_1 = f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}\n\nSelect the correct option."
run_answer_experiment("1. Baseline (常规基线 - 模拟当前故障情况)", sys_1, user_1)

# 方案 2: Recency Effect (近因效应 - 把选项贴脸放在长文本最后)
user_2 = f"Read this context carefully:\n{CONTEXT}\n\nNow answer this question:\n{QUESTION}\n\nOutput ONLY the uppercase letter (A, B, C, D, or E) of the correct option."
run_answer_experiment("2. Recency Effect (选项贴脸 + 强约束)", sys_1, user_2)

# 方案 3: Strict System Constraint (极简强硬规则)
sys_3 = "You are an expert medical exam taker. Output EXACTLY ONE uppercase letter (A, B, C, D, or E). Do not output anything else."
run_answer_experiment("3. Strict System Constraint (强制只输出字母)", sys_3, user_2)

# 方案 4: CoT (让它在长文本中先提取关键句，再做决定)
sys_4 = "You are an expert. First, extract the single sentence from the Context that answers the question. Then state 'Final Answer: [Letter]'."
run_answer_experiment("4. CoT (先提取证据后输出)", sys_4, user_2, max_tokens=150)

# 方案 5: Negative Constraint (严禁胡说八道)
sys_5 = "Select the correct option. CRITICAL: Do NOT output the text of the option. Do NOT explain. ONLY output the single letter A, B, C, D, or E."
run_answer_experiment("5. Negative Constraint (严禁输出多余文字)", sys_5, user_2)

# 方案 6: Greedy Decoding (绝对零度)
run_answer_experiment("6. Greedy Decoding (绝对零度)", sys_3, user_2, temp=0.0)

# 方案 7: 纯粹对比指令 (不提医学，只做连连看)
sys_7 = "Compare the facts in the Context against the Options. Which exact Option matches the Context? Output only the letter."
run_answer_experiment("7. Contrastive Matching (纯文本比对)", sys_7, user_2)