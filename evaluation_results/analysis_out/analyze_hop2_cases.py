import pandas as pd
import csv

PATH = "tf_with_simulated_policy.csv"  # 按你的实际路径改

# 关键：用 python csv 解析多行字段
rows = []
with open(PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
df = pd.DataFrame(rows)

def to_bool(x):
    s = str(x).strip().lower()
    if s in ("true","1","yes","y","t"): return True
    if s in ("false","0","no","n","f",""): return False
    return None

for c in ["Hop2_Triggered_bool","Completion_Triggered_bool","Is_Correct_bool"]:
    if c in df.columns:
        df[c] = df[c].apply(to_bool)

tf = df[df["Type"].str.upper() == "TF"].copy()

# 只看 hop2 触发的
h2 = tf[tf["Hop2_Triggered_bool"] == True].copy()

# 统一取关键列
cols = [
    "Question_ID","Is_Correct","Generated_Answer","Ground_Truth_Answer",
    "TF_NLI_Label_Pre","TF_NLI_Label_Hop1","TF_NLI_Label_Hop2","TF_Final_NLI_Label_Used",
    "Web_Query_Used_Hop1","Web_Query_Used_Hop2","Hop2_Reason",
    "TF_NLI_Confidence_Pre","TF_NLI_Confidence_Hop2",
]

for c in cols:
    if c not in h2.columns:
        h2[c] = ""

# 1) hop2 翻盘：pre NEI -> final SUPPORTED
win = h2[(h2["TF_NLI_Label_Pre"] == "NOT_ENOUGH_INFO") &
         (h2["TF_Final_NLI_Label_Used"] == "SUPPORTED")].copy()

# 2) hop2 未翻盘：final 仍 NEI
stuck = h2[h2["TF_Final_NLI_Label_Used"] == "NOT_ENOUGH_INFO"].copy()

# 3) 可疑：hop2 后 final 变成 CONTRADICTED（或任何非 SUPPORTED/NEI）
susp = h2[~h2["TF_Final_NLI_Label_Used"].isin(["SUPPORTED","NOT_ENOUGH_INFO"])].copy()

win[cols].to_csv("hop2_win_list.csv", index=False)
stuck[cols].to_csv("hop2_stuck_list.csv", index=False)
susp[cols].to_csv("hop2_suspicious_list.csv", index=False)

print("Hop2 triggered N =", len(h2))
print("Hop2 win (NEI->SUPPORTED) =", len(win))
print("Hop2 stuck (final NEI) =", len(stuck))
print("Hop2 suspicious (other final labels) =", len(susp))
