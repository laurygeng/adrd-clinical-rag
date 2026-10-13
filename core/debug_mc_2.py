import os
import time
import re
import json
from llm_engine import LocalLLMClient

# 动态获取环境变量中的模型
model_id = os.environ.get("GLOBAL_AGENT_MODEL")
if not model_id:
    print("⚠️ 未检测到 GLOBAL_AGENT_MODEL 环境变量，使用默认模型作兜底。")
    model_id = "ZJU-AI4H/Lingshu_7B"

client = LocalLLMClient()

print(f"🚀 开始测试终极 JSON CoT 提纯方案...")
print(f"🤖 当前测试模型: {model_id}\n")

# =========================================================================
# 场景 A: 干扰项陷阱测试 (目标: 拦截通用科普废话)
# =========================================================================
QUERY_A = "How should out-of-season clothing be handled for dementia patients to reduce confusion?"
NOISE_TEXT_A = """Alzheimer's Disease dementia care guidelines clothing organization. 
Individuals with Alzheimer's may experience difficulty with organizing their clothing due to memory loss and confusion. 
Clothing organization techniques such as using labels, color-coding, and placing items in clear plastic bins can help individuals maintain independence. 
It is important to adapt clothing organization systems based on the individual's specific needs."""

# =========================================================================
# 场景 B: 幻觉与复读机测试 (目标: 无答案时输出 NONE)
# =========================================================================
QUERY_B = "What is the mandatory lighting requirement for the patient's room during the day?"
NOISE_TEXT_B = """Delirium Management: The text does not provide any specific information about lighting requirements for a patient room or how they relate to dementia or delirium. It also does not mention daytime care."""

def extract_json_value(txt):
    """模拟生产环境的安全 JSON 提取与后处理"""
    txt = re.split(r"(?i)### response", txt)[0].strip()
    match = re.search(r"\{.*?\}", txt, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return json.dumps(data, ensure_ascii=False, indent=2)
        except:
            return txt
    return txt

def run_cot_experiment(name, sys_prompt, user_prompt_template):
    print(f"{'='*80}\n🧪 方案: {name}")
    # 启用防复读超参
    os.environ["HF_REPETITION_PENALTY"] = "1.15"
    
    for scene_name, query, text in [("Scene A (干扰项测试)", QUERY_A, NOISE_TEXT_A), ("Scene B (无答案测试)", QUERY_B, NOISE_TEXT_B)]:
        user_prompt = user_prompt_template.format(query=query, text=text)
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=200
            )
            out_txt = response.choices[0].message.content.strip()
            clean_json = extract_json_value(out_txt)
            
            print(f"  [{scene_name}] ⏱️ {time.time()-t0:.2f}s")
            print(f"  📤 解析结果:\n{clean_json}\n")
            
            if "NONE" not in clean_json.upper():
                print("  ⚠️ 警告: 未输出 NONE，可能提取了通用废话或提取失败！")
        except Exception as e:
            print(f"  ❌ [{scene_name}] 运行失败: {e}")

# -------------------------------------------------------------------------
# 3 种 CoT JSON 变体 (专门针对负面约束盲区)
# -------------------------------------------------------------------------

sys_base = "You are a clinical data parser. Output valid JSON ONLY. Do not converse."

# 方案 1: 布尔校验 (Boolean Verification)
user_1 = """Task:
1. Determine if the text specifically answers the exact clinical scenario in the query. If it only provides broad, generic guidelines, set "is_exact_match" to false.
2. If "is_exact_match" is true, extract the fact. If false, output "NONE".

Query: '{query}'
Text: {text}

Format: {{"is_exact_match": true/false, "extracted": "fact or NONE"}}
JSON:"""
run_cot_experiment("1. CoT Boolean (前置布尔校验)", sys_base, user_1)

# 方案 2: 短推理域 (Short Reasoning Field)
user_2 = """Task: Output JSON with two keys:
- "reasoning": 1 short sentence evaluating if the text contains the SPECIFIC answer to the query, or if it is just generic background advice.
- "extracted": The specific clinical fact, or "NONE" if the text lacks the specific answer.

Query: '{query}'
Text: {text}

Format: {{"reasoning": "...", "extracted": "fact or NONE"}}
JSON:"""
run_cot_experiment("2. Short Reasoning (一句话推理)", sys_base, user_2)

# 方案 3: 内容分类标签 (Content Type Classification)
user_3 = """Task: Does the text answer the SPECIFIC query, or is it just generic advice?
Query: '{query}'
Text: {text}

Format:
{{
  "content_type": "GENERIC_ADVICE" or "SPECIFIC_ANSWER" or "NO_ANSWER",
  "extracted": "fact or NONE"
}}
JSON:"""
run_cot_experiment("3. Content Type Classification (打标分类)", sys_base, user_3)