# Local-KB Sufficiency Evaluator — Experiment Log

Goal: a **novel, self-contained** way to judge whether the **local KB** retrieved enough
to answer a question, and (when not) to auto-complete the missing knowledge — **without any
external paid tool/API at inference** and **without fine-tuning the generator**. The evaluator
is the retrieval-side contribution.

---

## 0. What the AUC actually measures

All AUC numbers below use the **same metric and the same proxy ground truth**:

> **AUC( evaluator's *sufficiency* signal  →  was the final answer CORRECT )**

- Ground-truth proxy: **`answer correct ⟺ context was sufficient`** (we have no human
  sufficient/insufficient labels, so this is the objective stand-in).
- AUC is **threshold-free** (0.5 = random, 1.0 = perfect); it scores *ranking quality*.
- **Test sets differ — do NOT compare across them blindly:**
  - prompt methods & **zero-shot** trained models → tested on all **149** questions (no ADRD
    training, so no leakage).
  - **ADRD-adapted** trained models → tested on a **held-out 49** (other 100 used for
    training); small ⇒ noisy.

The proxy is **noisy, especially for TF** (true/false has a 50% guess baseline), which caps
how high any method's AUC can look.

---

## 1. All sufficiency-evaluation methods tried

| Method | Kind | AUC | Test set |
|---|---|---|---|
| GPT-4o prompt (high/med/low confidence) | prompt | TF ~0.63 | 149 |
| NLI answerability (entail/contradict, `nli-deberta-v3-base`) | local model | TF 0.675 | 149 |
| **ItV (Identify-then-Verify)**, gpt-4o-mini, N=5 self-consistency | prompt | **TF 0.694 / MC 0.731** | 149 |
| CRAG released evaluator (T5-large), zero-shot | trained | 0.469 (failed to transfer) | 149 |
| Self-trained RoBERTa — **general** (SQuAD2+FEVER), zero-shot | trained | TF 0.519 / MC 0.564 / ALL 0.529 | 149 |
| Self-trained RoBERTa — **general+medical** (+PubMedQA), zero-shot | trained | TF 0.577 / **MC 0.692** / ALL 0.595 | 149 |
| Self-trained + ItV-adapted (**general** base) | trained | ALL 0.548 | 49 |
| Self-trained + ItV-adapted (**medical** base) | trained | **ALL 0.622** | 49 |

**Two "bests":**
- **Best overall (prompt):** ItV — **MC 0.731 / TF 0.694** on full 149. This is the bar to
  match/replace.
- **Best trained model:** medical base + ItV adaptation — **0.622** on held-out 49 (noisy);
  zero-shot best is the medical base **MC 0.692** on full 149.

---

## 2. The trained evaluator — what the best version does

A local classifier (`roberta-base`, input `(question, context)`, output `P(sufficient)`,
**one forward pass, zero external dependency at inference**), built in two stages.

### Stage 1 — general + medical pretraining  (`train_sufficiency.py` → `/tmp/suff_roberta_med`)
~20k examples with **naturally-available** sufficiency labels:
- **SQuAD2 (8k):** answerable → sufficient(1) / unanswerable → insufficient(0).
- **FEVER (4k):** SUPPORTS/REFUTES → sufficient(1) / NOT-ENOUGH-INFO → insufficient(0).
- **PubMedQA (8k):** a question's **own abstract → sufficient(1)** / a **swapped unrelated
  abstract → insufficient(0)**. ← medical grounding; lifted MC zero-shot 0.564 → 0.692.

1 epoch, CPU, MAXLEN 256, lr 2e-5.

### Stage 2 — ItV-distilled domain adaptation  (`adapt_sufficiency.py`)
- Start from the medical base.
- Build ADRD data from 100 train questions (49 held out), **labels distilled from the ItV
  teacher** (gpt-4o-mini, N=5): per question — full context → ItV label; degraded (2 random
  passages) → ItV label; swapped (another question's passages) → 0.
- Fine-tune 3 epochs, lr 1e-5, CPU.
- Held-out 49: **0.622**, beating the ItV teacher on that same split (0.548).

**Novelty framing:** a self-contained local evaluator via *general+medical pretraining →
ItV distillation*, aimed at replacing the prompt-based ItV at zero inference cost.

---

## 3. Honest status — not production-ready yet

A confusion-matrix read (`grade_evaluator.py`, medical base, threshold 0.5) exposes why AUC
alone flatters it:

```
        accuracy   said-ENOUGH&WRONG (missed)   said-not-enough&wrong (caught)
TF      0.700           14                            4
MC      0.862            3                            0
ALL     0.732           17                            4
```

- **Accuracy 0.732 is a trap:** 128/149 (86%) were answered correctly anyway, so a trivial
  "always say ENOUGH" scores **0.859** — higher than ours. Accuracy rewards the majority class.
- **The metric that matters for a GATE is the catch rate** (recall on wrong-answer questions):
  currently **0.19** (caught 4 of 21). It misses ~80% of the cases that actually need
  completion ⇒ **not usable in production yet**.
- AUC ≈ 0.6 says the *ranking signal* is weak; at threshold 0.5 it doesn't convert into useful
  decisions.

---

## 4. Improvement levers (future work, by expected payoff)

1. **Biomedical encoder as base** (highest payoff): replace `roberta-base` with
   **PubMedBERT / BioLinkBERT** (pretrained on PubMed) — medical grounding for free.
2. **Soft-label distillation:** train on ItV's continuous `none_frac` instead of hard 0/1.
3. **TF-specific help** (TF is the weak type): fuse the NLI decisive-entail/contradict feature,
   or add TF-targeted data.
4. **Threshold tuning** for the gate: misses (say-enough-but-wrong) are costly, false alarms
   (say-not-enough-but-right) only cost an extra search ⇒ set a **low** threshold to favor
   recall; sweep the catch-rate/false-alarm tradeoff.
5. **More adaptation data** (more context variants / ItV labels per question).

---

## 5. Files

| File | Role |
|---|---|
| `train_sufficiency.py` | Stage 1 — general+medical pretraining; zero-shot ADRD AUC report |
| `adapt_sufficiency.py` | Stage 2 — ItV-distilled ADRD adaptation; held-out AUC vs base/ItV |
| `grade_evaluator.py` | Confusion matrix + accuracy/catch-rate at a threshold (the §3 read) |
| `gate.py` | Live pipeline gate: NLI (TF) + ItV (MC) sufficiency, self-contained |
| `tf_gate_compare.py` | NLI-vs-ItV TF gate comparison |

---

## 6. Pipeline-side work (context)

Retrieval bug fixes (smart-window header strip), re-ingest with `bge-large-en-v1.5`,
BM25/dense = 0.65/0.35, `pre_k=30`, `bge-reranker-v2-m3`, sufficiency gate (TF=NLI / MC=ItV),
gap-guided **local** re-retrieval, free web fallback (PubMed + curated allowlist + junk-sentence
filter), strict-grounding generation. **Best end-to-end: 130/149 = 87.2% (130/144 = 90.3%
excluding 5 Bucket-B ambiguous items).** Of the referenced papers, **only CRAG's idea
transferred to a real gain (+3)**.
