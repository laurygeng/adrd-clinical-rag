#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze benchmark_eval_all_*.csv from entrypoints/run_online_pipeline.py

Robust to multiline fields (e.g., Retrieved_Context with newlines) by using
python's csv module with newline='' and proper quoting.

Outputs:
- summary_overall.md
- summary_overall.csv
- tf_nli_tables.md
- tf_nli_distributions.csv
- tf_flip_matrix_pre_to_final.csv
- completion_reason_counts.csv
- hop2_reason_counts.csv
- court_veto_summary.csv
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import pandas as pd


TF_LABELS_4WAY = [
    "SUPPORTED",
    "HARD_CONTRADICTION",
    "SOFT_CONTRADICTION",
    "NOT_ENOUGH_INFO",
]

# Deterministic mapping policy (per your Iter-3 description)
def tf_label_to_answer(label: str) -> str:
    """Return 'Yes' or 'No' under current TF policy."""
    if label == "SUPPORTED":
        return "Yes"
    # hard contradiction, soft contradiction, NEI, missing -> No
    return "No"


def read_multiline_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return pd.DataFrame(rows)


def to_bool(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n", ""):
        return False
    return None


def normalize_type(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


def safe_str(x):
    if x is None:
        return ""
    return str(x)


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t, g in df.groupby("Type", dropna=False):
        n = len(g)
        correct = g["Is_Correct_bool"].sum()
        acc = (correct / n) if n else 0.0
        out.append({"Type": t, "N": n, "Correct": int(correct), "Accuracy": acc})
    # overall
    n = len(df)
    correct = df["Is_Correct_bool"].sum()
    out.append({"Type": "ALL", "N": n, "Correct": int(correct), "Accuracy": (correct / n) if n else 0.0})
    return pd.DataFrame(out).sort_values(["Type"])


def label_dist(series: pd.Series, label_order=None) -> pd.DataFrame:
    c = series.fillna("").astype(str).replace({"nan": ""})
    cnt = c.value_counts(dropna=False)
    labels = label_order if label_order else list(cnt.index)
    rows = []
    total = int(cnt.sum()) if len(cnt) else 0
    for lab in labels:
        k = int(cnt.get(lab, 0))
        rows.append({"Label": lab, "Count": k, "Frac": (k / total) if total else 0.0})
    # include any unexpected labels
    for lab in cnt.index:
        if label_order and lab not in label_order:
            k = int(cnt.get(lab, 0))
            rows.append({"Label": lab, "Count": k, "Frac": (k / total) if total else 0.0})
    return pd.DataFrame(rows)


def flip_matrix(df: pd.DataFrame, pre_col: str, post_col: str, labels=None) -> pd.DataFrame:
    labels = labels or sorted(set(df[pre_col].dropna().unique()) | set(df[post_col].dropna().unique()))
    mat = pd.crosstab(df[pre_col], df[post_col], dropna=False)

    # ensure full label set rows/cols
    for r in labels:
        if r not in mat.index:
            mat.loc[r] = 0
    for c in labels:
        if c not in mat.columns:
            mat[c] = 0
    mat = mat.loc[labels, labels]
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to benchmark_eval_all_*.csv")
    ap.add_argument("--outdir", default="analysis_out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = read_multiline_csv(args.csv)

    # Basic normalization / typing
    for col in ["Completion_Triggered", "Hop2_Triggered", "Veto_Triggered", "TF_Support_Injected", "Is_Correct"]:
        if col in df.columns:
            df[col + "_bool"] = df[col].apply(to_bool)

    if "Type" in df.columns:
        df["Type"] = df["Type"].apply(normalize_type)

    if "Is_Correct_bool" not in df.columns and "Is_Correct" in df.columns:
        df["Is_Correct_bool"] = df["Is_Correct"].apply(to_bool)

    # ============ Table: overall / by type ============
    acc_tbl = compute_accuracy(df)
    acc_tbl.to_csv(os.path.join(args.outdir, "summary_overall.csv"), index=False)

    # Trigger rates by type
    trig_rows = []
    for t, g in df.groupby("Type", dropna=False):
        n = len(g)
        comp = g["Completion_Triggered_bool"].sum() if "Completion_Triggered_bool" in g else 0
        hop2 = g["Hop2_Triggered_bool"].sum() if "Hop2_Triggered_bool" in g else 0
        veto = g["Veto_Triggered_bool"].sum() if "Veto_Triggered_bool" in g else 0
        trig_rows.append({
            "Type": t,
            "N": n,
            "Completion_Triggered": int(comp),
            "Completion_Triggered_Rate": (comp / n) if n else 0.0,
            "Hop2_Triggered": int(hop2),
            "Hop2_Triggered_Rate": (hop2 / n) if n else 0.0,
            "Veto_Triggered": int(veto),
            "Veto_Triggered_Rate": (veto / n) if n else 0.0,
        })
    trig_tbl = pd.DataFrame(trig_rows).sort_values(["Type"])
    trig_tbl.to_csv(os.path.join(args.outdir, "trigger_rates_by_type.csv"), index=False)

    # Reasons
    if "Completion_Reason" in df.columns:
        cr = df["Completion_Reason"].fillna("").astype(str)
        cr_cnt = cr[cr != ""].value_counts().reset_index()
        cr_cnt.columns = ["Completion_Reason", "Count"]
        cr_cnt.to_csv(os.path.join(args.outdir, "completion_reason_counts.csv"), index=False)

    if "Hop2_Reason" in df.columns:
        hr = df["Hop2_Reason"].fillna("").astype(str)
        hr_cnt = hr[hr != ""].value_counts().reset_index()
        hr_cnt.columns = ["Hop2_Reason", "Count"]
        hr_cnt.to_csv(os.path.join(args.outdir, "hop2_reason_counts.csv"), index=False)

    # ============ TF-specific NLI tables ============
    tf = df[df["Type"] == "TF"].copy()
    tf_out_md = []

    tf_out_md.append("# TF NLI Analysis\n")
    tf_out_md.append(f"- Total TF questions: {len(tf)}\n")

    for col in ["TF_NLI_Label_Pre", "TF_NLI_Label_Hop1", "TF_NLI_Label_Hop2", "TF_Final_NLI_Label_Used"]:
        if col in tf.columns:
            dist = label_dist(tf[col], label_order=TF_LABELS_4WAY + [""])
            dist.to_csv(os.path.join(args.outdir, f"tf_nli_dist_{col}.csv"), index=False)
            tf_out_md.append(f"## Label distribution: `{col}`\n")
            tf_out_md.append(dist.to_markdown(index=False))
            tf_out_md.append("\n")

    # Flip matrix pre->final
    if "TF_NLI_Label_Pre" in tf.columns and "TF_Final_NLI_Label_Used" in tf.columns:
        pre = tf["TF_NLI_Label_Pre"].fillna("").astype(str)
        fin = tf["TF_Final_NLI_Label_Used"].fillna("").astype(str)
        tf["pre_label"] = pre
        tf["final_label"] = fin

        mat = flip_matrix(tf, "pre_label", "final_label", labels=TF_LABELS_4WAY + [""])
        mat.to_csv(os.path.join(args.outdir, "tf_flip_matrix_pre_to_final.csv"))
        tf_out_md.append("## Flip matrix: pre → final\n")
        tf_out_md.append(mat.to_markdown())
        tf_out_md.append("\n")

        flip_rate = (tf["pre_label"] != tf["final_label"]).mean() if len(tf) else 0.0
        tf_out_md.append(f"- Flip rate (pre != final): {flip_rate:.3f}\n")

    # Completion/hop2 contributions (simulate “pre-only policy” vs final policy)
    # Uses current deterministic mapping for TF.
    if len(tf) and "TF_NLI_Label_Pre" in tf.columns and "TF_Final_NLI_Label_Used" in tf.columns:
        tf["pred_answer_pre_only"] = tf["TF_NLI_Label_Pre"].apply(lambda x: tf_label_to_answer(safe_str(x).strip()))
        tf["pred_answer_final"] = tf["TF_Final_NLI_Label_Used"].apply(lambda x: tf_label_to_answer(safe_str(x).strip()))

        # Compare to Ground_Truth_Answer (Yes/No)
        gt = tf["Ground_Truth_Answer"].fillna("").astype(str).str.strip()
        tf["gt"] = gt

        tf["pre_only_correct_sim"] = (tf["pred_answer_pre_only"] == tf["gt"])
        tf["final_correct_sim"] = (tf["pred_answer_final"] == tf["gt"])

        # NOTE: This is a *policy simulation* independent of Generated_Answer.
        # If you want to use actual pipeline correctness, use Is_Correct_bool.
        sim_pre_acc = tf["pre_only_correct_sim"].mean()
        sim_final_acc = tf["final_correct_sim"].mean()
        sim_gain = sim_final_acc - sim_pre_acc

        tf_out_md.append("## TF policy simulation (deterministic mapping)\n")
        tf_out_md.append(f"- Simulated acc using pre-label only: {sim_pre_acc:.3f}\n")
        tf_out_md.append(f"- Simulated acc using final-label used: {sim_final_acc:.3f}\n")
        tf_out_md.append(f"- Simulated gain (final - pre): {sim_gain:.3f}\n")

        # Net gain among completion-triggered
        if "Completion_Triggered_bool" in tf.columns:
            comp = tf[tf["Completion_Triggered_bool"] == True]
            if len(comp):
                pre_acc_c = comp["pre_only_correct_sim"].mean()
                fin_acc_c = comp["final_correct_sim"].mean()
                tf_out_md.append("\n### Among Completion_Triggered=True\n")
                tf_out_md.append(f"- N={len(comp)} pre-only acc={pre_acc_c:.3f} final acc={fin_acc_c:.3f} gain={fin_acc_c - pre_acc_c:.3f}\n")

        if "Hop2_Triggered_bool" in tf.columns:
            hop2 = tf[tf["Hop2_Triggered_bool"] == True]
            if len(hop2):
                pre_acc_h = hop2["pre_only_correct_sim"].mean()
                fin_acc_h = hop2["final_correct_sim"].mean()
                tf_out_md.append("\n### Among Hop2_Triggered=True\n")
                tf_out_md.append(f"- N={len(hop2)} pre-only acc={pre_acc_h:.3f} final acc={fin_acc_h:.3f} gain={fin_acc_h - pre_acc_h:.3f}\n")

        tf.to_csv(os.path.join(args.outdir, "tf_with_simulated_policy.csv"), index=False)

    with open(os.path.join(args.outdir, "tf_nli_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(tf_out_md))

    # ============ Court/Veto summary ============
    court_rows = []
    if "Veto_Triggered_bool" in df.columns:
        veto_true = df[df["Veto_Triggered_bool"] == True]
        veto_false = df[df["Veto_Triggered_bool"] == False]

        def acc_of(d):
            return d["Is_Correct_bool"].mean() if len(d) else 0.0

        court_rows.append({"Group": "Veto_Triggered=True", "N": len(veto_true), "Accuracy(Is_Correct)": acc_of(veto_true)})
        court_rows.append({"Group": "Veto_Triggered=False", "N": len(veto_false), "Accuracy(Is_Correct)": acc_of(veto_false)})

        # verdict distribution if exists
        if "Court_Verdict" in df.columns:
            vd = df["Court_Verdict"].fillna("").astype(str)
            vd_cnt = vd[vd != ""].value_counts().reset_index()
            vd_cnt.columns = ["Court_Verdict", "Count"]
            vd_cnt.to_csv(os.path.join(args.outdir, "court_verdict_counts.csv"), index=False)

    pd.DataFrame(court_rows).to_csv(os.path.join(args.outdir, "court_veto_summary.csv"), index=False)

    # ============ Write a top-level markdown summary ============
    md = []
    md.append("# Benchmark Evaluation Summary\n")
    md.append("## Accuracy by Type\n")
    md.append(acc_tbl.to_markdown(index=False))
    md.append("\n## Trigger rates by Type\n")
    md.append(trig_tbl.to_markdown(index=False))

    with open(os.path.join(args.outdir, "summary_overall.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"[OK] Wrote analysis outputs to: {args.outdir}")
    print(f"- {os.path.join(args.outdir, 'summary_overall.md')}")
    print(f"- {os.path.join(args.outdir, 'tf_nli_tables.md')}")


if __name__ == "__main__":
    main()
