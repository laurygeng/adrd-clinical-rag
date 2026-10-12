import os
import time
from llm_engine import LocalLLMClient

# 靶子题目：ADRD_MC_012 (洗澡抗拒题)
QUESTION = """A caregiver is assisting a patient living with dementia who is resisting a bath. What is the primary recommended action if the resistance becomes significant?

Options:
  A. Call another family member to help physically guide the patient into the shower.
  B. Wait a moment and try to calmly explain to the patient the importance of bathing for skin health to gain their cooperation.
  C. Offer a favorite snack as a reward for completing the bath.
  D. Explain the health risks of poor hygiene to persuade the patient.
  E. Skip the bath for today and try again tomorrow if possible."""

# 隐藏选项的版本（方案2用）
QUESTION_NO_OPTIONS = "A caregiver is assisting a patient living with dementia who is resisting a bath. What is the primary recommended action if the resistance becomes significant?"

CONTEXT = """Bathing assistance in dementia patients can enhance their quality of life by reducing resistance. Every individual with dementia has a distinct approach required due to their uniqueness. Clinical Fact: Significant resistance may occur when managing dementia care, particularly with bathing refusal. Providing assistance and support is crucial in such situations."""

# 初始化 LLM 客户端
client = LocalLLMClient()
model_id = os.environ.get("GLOBAL_AGENT_MODEL", "ZJU-AI4H/Lingshu_7B")

print(f"🚀 开始测试 Lingshu_7B 的 10 维极限方案...\n")

def run_experiment(name, sys_prompt, user_prompt, temp=0.3, max_tokens=80, rep=1.05, prefill=""):
    print(f"{'='*60}")
    print(f"🧪 方案: {name}")
    try:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
            
        t0 = time.time()
        # 这里动态覆盖底层的一些参数用于测试
        os.environ["HF_REPETITION_PENALTY"] = str(rep)
        
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens
        )
        text = response.choices[0].message.content.strip()
        print(f"⏱️ 耗时: {time.time()-t0:.2f}s")
        print(f"📤 输出:\n{text}")
    except Exception as e:
        print(f"❌ 运行失败: {e}")
    print(f"{'='*60}\n")


# -------------------------------------------------------------------------
# 10 种极限变异方案
# -------------------------------------------------------------------------

# 方案 1: Baseline (当前基线，用于对比)
sys_1 = "You are an evaluator. Determine if the context contains enough info to answer the question. Output valid JSON: {\"sufficient\": true/false, \"gap\": \"missing info or NONE\"}"
run_experiment("1. Baseline (当前基线)", sys_1, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}")

# 方案 2: Option Masking (屏蔽选项法 - 彻底杜绝它寻找ABCDE)
run_experiment("2. Option Masking (隐藏选项)", sys_1, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION_NO_OPTIONS}")

# 方案 3: Chain-of-Thought JSON (思维链强制输出 - 让它先思考再给结论)
sys_3 = "Output JSON. First provide 'reasoning' step-by-step, then 'sufficient', then 'gap'. Format: {\"reasoning\": \"...\", \"sufficient\": false, \"gap\": \"...\"}"
run_experiment("3. CoT JSON (强制思维链)", sys_3, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}")

# 方案 4: YAML / Plain Text (抛弃 JSON - 避开它的括号恐惧症)
sys_4 = "Do NOT output JSON. Output EXACTLY two lines:\nsufficient: [true/false]\ngap: [NONE or short description]"
run_experiment("4. Plain Text (抛弃 JSON 格式)", sys_4, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}")

# 方案 5: Greedy Decoding (绝对零度解码 - 消除一切随机性)
run_experiment("5. Greedy Decoding (温度 0.0)", sys_1, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}", temp=0.0)

# 方案 6: High Repetition Penalty (极高重复惩罚 - 专治复读机)
run_experiment("6. High RepPenalty=1.5 (极高重复惩罚)", sys_1, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}", rep=1.5)

# 方案 7: Few-Shot (少样本对齐 - 给它抄作业)
sys_7 = """You are an evaluator. Output JSON.
Example:
Context: Patient is thirsty.
Question: What should caregiver do?
Output: {"sufficient": false, "gap": "specific action to alleviate thirst"}

Now do this one:"""
run_experiment("7. Few-Shot (少样本强行对齐)", sys_7, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}")

# 方案 8: Prefix Tuning (强制补全开头)
sys_8 = "Determine if context is sufficient. Output JSON: {\"sufficient\": false, \"gap\": \"...\"}"
run_experiment("8. Prefix Tuning (强制喂入前缀)", sys_8, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}", prefill="{\n  \"sufficient\": false,\n  \"gap\": \"")

# 方案 9: Keyword Only (纯粹关键词 - 越短越不容易错)
sys_9 = "If context is sufficient, output NONE. If insufficient, output EXACTLY 3 keywords representing the missing medical action. DO NOT output sentences."
run_experiment("9. Keyword Only (仅输出3个关键词)", sys_9, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}")

# 方案 10: Clinical Persona (换脑法 - 唤醒它的医生人格)
sys_10 = "You are a Senior Clinical Reviewer. Does the context contain the specific medical intervention needed? Output JSON: {\"sufficient\": true/false, \"gap\": \"missing intervention\"}"
run_experiment("10. Clinical Persona (唤醒医生人格)", sys_10, f"Context:\n{CONTEXT}\n\nQuestion:\n{QUESTION}")