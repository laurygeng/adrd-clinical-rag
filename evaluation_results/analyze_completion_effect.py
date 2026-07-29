import pandas as pd

CSV_PATH = "benchmark_eval_all_20260727_013849.csv"

df = pd.read_csv(CSV_PATH)

def acc(x):
    return float(x.mean()) if len(x) else float("nan")

# Normalize booleans (some CSVs may store as True/False strings)
if df["Is_Correct"].dtype == object:
    df["Is_Correct"] = df["Is_Correct"].astype(str).str.lower().map({"true": True, "false": False})

# Basic overall
print("=== Overall ===")
print("N:", len(df))
print("Accuracy:", acc(df["Is_Correct"]))

# By type
print("\n=== By Type ===")
for t, g in df.groupby("Type"):
    print(t, "N:", len(g), "Acc:", acc(g["Is_Correct"]))

# Two groups you requested
grp_a = df[(df["Completion_Triggered"] == True) & (df["Veto_Triggered"] == False)]
grp_b = df[(df["Completion_Triggered"] == False)]

print("\n=== Completion Accepted (Completion_Triggered=True & Veto_Triggered=False) ===")
print("N:", len(grp_a), "Acc:", acc(grp_a["Is_Correct"]))

print("\n=== No Completion (Completion_Triggered=False) ===")
print("N:", len(grp_b), "Acc:", acc(grp_b["Is_Correct"]))

# Same two groups but restricted to TF (since TF is where you see the issue)
tf = df[df["Type"] == "TF"]
tf_a = tf[(tf["Completion_Triggered"] == True) & (tf["Veto_Triggered"] == False)]
tf_b = tf[(tf["Completion_Triggered"] == False)]

print("\n=== TF: Completion Accepted ===")
print("N:", len(tf_a), "Acc:", acc(tf_a["Is_Correct"]))

print("\n=== TF: No Completion ===")
print("N:", len(tf_b), "Acc:", acc(tf_b["Is_Correct"]))

# Optional: show the worst offenders where completion was accepted but answer is wrong
bad = tf_a[tf_a["Is_Correct"] == False][
    ["Question_ID", "Generated_Answer", "Ground_Truth_Answer", "Web_Query_Used", "Critic_Missing_Info", "Critic_MD_Path"]
]
print("\n=== TF: Wrong but Completion Accepted (sample) ===")
print(bad.head(30).to_string(index=False))
