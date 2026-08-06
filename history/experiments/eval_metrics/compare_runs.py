import pandas as pd

files = [
    ("无RAG (旧)",      "generate/answers/answers_gemini3_flash_ADRD_all_rag_20260601_102628.csv"),
    ("RAG旧检索",       "generate/answers/answers_gemini3_flash_ADRD_all_rag_20260601_123019.csv"),
    ("RAG新检索+外网",  "generate/answers/answers_gemini3_flash_ADRD_all_rag_20260602_112453.csv"),
]

header = "{:<22} {:>10} {:>10} {:>10}".format("版本", "MC(29)", "TF(120)", "Overall")
print(header)
print("-" * 54)
for label, path in files:
    try:
        df = pd.read_csv(path)
        mc = df[df["Type"] == "MC"]
        tf = df[df["Type"] == "TF"]
        mc_acc = mc["Is_Correct"].sum() / len(mc) * 100
        tf_acc = tf["Is_Correct"].sum() / len(tf) * 100
        ov_acc = df["Is_Correct"].sum()  / len(df) * 100
        print("{:<22} {:>9.1f}% {:>9.1f}% {:>9.1f}%".format(label, mc_acc, tf_acc, ov_acc))
    except Exception as e:
        print("{:<22} 读取失败: {}".format(label, e))
