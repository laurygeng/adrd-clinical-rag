import os
import time
import re
import pandas as pd
from llm_engine import LocalLLMClient

# === 配置区 ===
MC_CSV_PATH = 'benchmark_eval_mc_lingshu_medical_mllm_Lingshu_7B_20261012_234852.csv'
TF_CSV_PATH = 'benchmark_eval_tf_dmis_lab_llama_3_meerkat_8b_v1.0_20261013_012154.csv' 

model_id = os.environ.get("GLOBAL_AGENT_MODEL")
if not model_id:
    raise ValueError("❌ 未检测到模型！请先设置环境变量，例如: export GLOBAL_AGENT_MODEL='dmis-lab/llama-3-meerkat-8b-v1.0'")

client = LocalLLMClient()

print(f"🚀 开始全量运行 RAG 上下文重组排序消融实验...")
print(f"🤖 当前测试模型: {model_id}\n")

# =====================================================================
# 1. 工具函数：拆解合并 Context
# =====================================================================
def extract_blocks(passages_text):
    text = str(passages_text)
    base_blocks = re.findall(r'(\[BASE\].*?)(?=\n\[(?:BASE|GAP|WEB)|$)', text, re.DOTALL)
    gap_blocks = re.findall(r'(\[GAP\].*?)(?=\n\[(?:BASE|GAP|WEB)|$)', text, re.DOTALL)
    web_blocks = re.findall(r'(\[WEB.*?\].*?)(?=\n\[(?:BASE|GAP|WEB)|$)', text, re.DOTALL)
    return base_blocks, gap_blocks, web_blocks

def assemble(b_blocks, g_blocks, w_blocks, strategy_idx):
    if strategy_idx == 1: return "\n\n".join(g_blocks + w_blocks + b_blocks)
    elif strategy_idx == 2: return "\n\n".join(b_blocks + g_blocks + w_blocks)
    elif strategy_idx == 3: return "\n\n".join(b_blocks[:1] + g_blocks + w_blocks + b_blocks[1:])
    elif strategy_idx == 4: return "\n\n".join(w_blocks + g_blocks + b_blocks)
    elif strategy_idx == 5: return "\n\n".join(g_blocks + b_blocks + w_blocks)
    return ""

# =====================================================================
# 2. 全量评测执行
# =====================================================================
def run_ab_test():
    results_list = []
    strategies = {
        1: "GAP -> WEB -> BASE (原TF巅峰策略 - 缺口优先)",
        2: "BASE -> GAP -> WEB (纯底座优先)",
        3: "BASE[0] -> GAP -> WEB -> BASE[Rest] (三明治夹心策略)",
        4: "WEB -> GAP -> BASE (外网新知识绝对优先)",
        5: "GAP -> BASE -> WEB (缺口打头外网垫底)"
    }

    files_to_test = [("MC", MC_CSV_PATH), ("TF", TF_CSV_PATH)]
    
    for q_type, filepath in files_to_test:
        if not os.path.exists(filepath):
            print(f"⚠️ 找不到文件: {filepath}")
            continue
            
        df = pd.read_csv(filepath)
        # 提取错题且包含检索上下文的记录
        wrong_df = df[(df['Is_Correct'] == False) & (df['Retrieved_Passages'].notna())]
        print(f"\n📦 正在测试 {q_type} 题库中的 {len(wrong_df)} 道错题...")

        if q_type == "MC":
            sys_prompt = "You are an expert medical exam taker. Output EXACTLY ONE uppercase letter (A, B, C, D, or E). Do not output anything else."
            user_tmpl = "Read this context carefully:\n{context}\n\nNow answer this question:\n{question}\n\nOutput ONLY the uppercase letter (A, B, C, D, or E)."
        else:
            sys_prompt = "You are an expert medical exam taker. Output EXACTLY Yes or No. Do not output anything else."
            user_tmpl = "Read this context carefully:\n{context}\n\nNow evaluate this statement:\n{question}\n\nOutput ONLY Yes or No."

        for _, row in wrong_df.iterrows():
            q_id = row['Question_ID']
            gt = str(row['Ground_Truth_Answer']).strip().upper()
            b, g, w = extract_blocks(row['Retrieved_Passages'])
            
            if not (len(b) > 0 and (len(g) > 0 or len(w) > 0)):
                continue # 没有拼装组合条件，测排序无意义跳过
            
            print(f"🎯 正在推理: [{q_id}]")
            row_data = {"Question_ID": q_id, "Type": q_type, "GT": gt}
            
            for s_idx, s_name in strategies.items():
                context = assemble(b, g, w, s_idx)
                if not context: continue
                
                user_prompt = user_tmpl.format(context=context, question=row['Question'])
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                        temperature=0.0,
                        max_tokens=10
                    )
                    ans = response.choices[0].message.content.strip().upper()
                    
                    # 匹配逻辑：TF 找 YES/NO，MC 找单字母
                    if q_type == "TF":
                        is_correct = 1 if (gt in ans or (gt == "YES" and "TRUE" in ans) or (gt == "NO" and "FALSE" in ans)) else 0
                    else:
                        match = re.search(r'\b([A-E])\b', ans)
                        parsed_ans = match.group(1) if match else ans
                        is_correct = 1 if parsed_ans == gt else 0
                        
                    row_data[f"Strategy_{s_idx}"] = is_correct
                except Exception as e:
                    row_data[f"Strategy_{s_idx}"] = 0
                    print(f"  ⚠️ 推理失败: {e}")
            
            results_list.append(row_data)

    # =====================================================================
    # 3. 统计与导出
    # =====================================================================
    if results_list:
        res_df = pd.DataFrame(results_list)
        res_df.to_csv("context_ablation_results.csv", index=False)
        print("\n✅ 所有错题遍历完毕！结果已导出至 context_ablation_results.csv")
        
        print("\n📊 排序策略错题挽救率统计 (数值代表将错题变对的比例):")
        for s_idx, s_name in strategies.items():
            col = f"Strategy_{s_idx}"
            mc_subset = res_df[res_df['Type'] == 'MC']
            tf_subset = res_df[res_df['Type'] == 'TF']
            
            mc_acc = mc_subset[col].mean() * 100 if len(mc_subset) > 0 else 0
            tf_acc = tf_subset[col].mean() * 100 if len(tf_subset) > 0 else 0
            
            print(f"[{s_name}]")
            print(f"  -> MC 挽救率: {mc_acc:.1f}%")
            print(f"  -> TF 挽救率: {tf_acc:.1f}%")

if __name__ == "__main__":
    run_ab_test()