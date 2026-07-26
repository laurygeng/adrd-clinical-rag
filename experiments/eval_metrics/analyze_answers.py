import pandas as pd, json

ans = pd.read_csv('generate/answers/answers_gemini3_flash_ADRD_all_rag_20260602_112453.csv')
ret = pd.read_csv('retrieval_results/retrieval_ADRD_all_k3_w500_20260602_091041.csv')
ret_map = {r['Question_ID']: dict(r) for _, r in ret.iterrows()}

wrong = ans[ans['Is_Correct'] == False]
right = ans[ans['Is_Correct'] == True]
print(f"答对: {len(right)} | 答错: {len(wrong)} | 准确率: {len(right)/len(ans)*100:.1f}%")
print()

# 拒答/无法回答检测
refusal_kws = ['none of the', 'cannot answer', 'not enough', 'no context',
               'cannot determine', 'insufficient', 'i cannot', 'not provided']
refusals = ans[ans['Generated_Answer'].str.lower().str.contains('|'.join(refusal_kws), na=False)]
print(f"拒答/无法回答题数: {len(refusals)}")
print()

# 答错题按 Satisfied 分组
wrong_ids = set(wrong['Question_ID'])
right_ids = set(right['Question_ID'])

sat_wrong  = sum(1 for qid in wrong_ids if ret_map.get(qid, {}).get('Satisfied') == True)
unsat_wrong = sum(1 for qid in wrong_ids if ret_map.get(qid, {}).get('Satisfied') == False)
sat_right  = sum(1 for qid in right_ids if ret_map.get(qid, {}).get('Satisfied') == True)
unsat_right = sum(1 for qid in right_ids if ret_map.get(qid, {}).get('Satisfied') == False)

print("=== 答错题的检索满足情况 ===")
print(f"答错 & 检索 Satisfied=True:  {sat_wrong}")
print(f"答错 & 检索 Satisfied=False: {unsat_wrong}")
print(f"答对 & 检索 Satisfied=True:  {sat_right}")
print(f"答对 & 检索 Satisfied=False: {unsat_right}")
print()

# Satisfied=True 时的准确率 vs Satisfied=False
sat_total   = sat_wrong + sat_right
unsat_total = unsat_wrong + unsat_right
if sat_total:
    print(f"检索 Satisfied=True  题目准确率: {sat_right}/{sat_total} = {sat_right/sat_total*100:.1f}%")
if unsat_total:
    print(f"检索 Satisfied=False 题目准确率: {unsat_right}/{unsat_total} = {unsat_right/unsat_total*100:.1f}%")
print()

# 打印前5道答错题详情
print("=== 答错题样例 (前5道) ===")
for i, (_, row) in enumerate(wrong.head(5).iterrows()):
    qid = row['Question_ID']
    rr  = ret_map.get(qid, {})
    srcs = json.loads(rr['Retrieved_Sources']) if rr else []
    web_c = sum(1 for s in srcs if str(s).startswith(('Wikipedia:','PubMed:','S2:')))
    sat   = rr.get('Satisfied', 'N/A') if rr else 'N/A'
    print(f"[{i+1}] {qid} | Type:{row['Type']} | Satisfied:{sat} | 本地:{len(srcs)-web_c} 外网:{web_c}")
    print(f"     Q: {str(row['Question'])[:110]}")
    print(f"     正确答案: {row['Ground_Truth_Answer']}")
    print(f"     模型输出: {str(row['Generated_Answer'])[:200]}")
    print()
