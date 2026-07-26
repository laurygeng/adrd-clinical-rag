import pandas as pd, json, numpy as np

old = pd.read_csv('retrieval_results/retrieval_ADRD_all_k3_w500_20260602_051939.csv')
new = pd.read_csv('retrieval_results/retrieval_ADRD_all_k3_w500_20260602_091041.csv')

def count_web(df):
    total, per_q = 0, []
    for _, row in df.iterrows():
        srcs = json.loads(row['Retrieved_Sources'])
        wc = sum(1 for s in srcs if str(s).startswith(('Wikipedia:','PubMed:','S2:')))
        total += wc
        per_q.append(wc)
    return total, per_q

def pcount(df):
    return [len(json.loads(r['Retrieved_Passages'])) for _, r in df.iterrows()]

def web_scores(df):
    s = []
    for _, row in df.iterrows():
        srcs   = json.loads(row['Retrieved_Sources'])
        scores = json.loads(row['Rerank_Scores'])
        for src, sc in zip(srcs, scores):
            if str(src).startswith(('Wikipedia:','PubMed:','S2:')):
                s.append(float(sc))
    return np.array(s) if s else np.array([0.0])

old_sat = old['Satisfied'].sum()
new_sat = new['Satisfied'].sum()
ow, op  = count_web(old)
nw, np_ = count_web(new)
op2, np2 = pcount(old), pcount(new)
os_, ns_ = web_scores(old), web_scores(new)

rows = [
    ("Satisfied 题数",         old_sat,                          new_sat),
    ("满足率 (%)",              round(old_sat/len(old)*100,1),    round(new_sat/len(new)*100,1)),
    ("外网 passages 总数",      ow,                               nw),
    ("用到外网的题数",           sum(1 for x in op if x>0),       sum(1 for x in np_ if x>0)),
    ("外网题平均 passages 数",   round(np.mean([x for x in op  if x>0]),1),
                                round(np.mean([x for x in np_ if x>0]),1)),
    ("每题平均 passages 总数",   round(np.mean(op2),1),           round(np.mean(np2),1)),
    ("passages > 30 的题数",    sum(1 for x in op2 if x>30),     sum(1 for x in np2 if x>30)),
    ("外网 passage 平均 score", round(float(os_.mean()),3),       round(float(ns_.mean()),3)),
    ("外网 score>=0.5 占比 (%)", round(sum(os_>=0.5)/len(os_)*100,1),
                                 round(sum(ns_>=0.5)/len(ns_)*100,1)),
    ("外网 score<0.3 占比 (%)",  round(sum(os_<0.3)/len(os_)*100,1),
                                 round(sum(ns_<0.3)/len(ns_)*100,1)),
]

print(f"\n{'指标':<30} {'旧结果(0051939)':>18} {'新结果(091041)':>16}")
print("-" * 66)
for label, ov, nv in rows:
    arrow = "↑" if nv > ov else ("↓" if nv < ov else "=")
    print(f"{label:<30} {str(ov):>18} {str(nv):>14} {arrow}")
print()
