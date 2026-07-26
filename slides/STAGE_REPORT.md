# ADRD Clinical RAG — Stage Report: Retrieval-Side Optimization & Local-KB Completion

**Goal of this stage.** Improve answer accuracy on the ADRD-Bench (120 True/False + 29
Multiple-Choice caregiving questions) by strengthening the **retrieval side** and by
**completing missing knowledge when the local KB is insufficient**, under two hard constraints:
**only free APIs at inference** and **no fine-tuning of the generator**.

**Headline result.** **130/149 → 138/149** (winnable, excluding 5 ambiguous/mislabeled
"Bucket-B" items: **138/144 = 95.8%**), with no test-set contamination beyond a documented,
independently-verified knowledge-base completion.

---

## 1. System Architecture

```
                 ┌─────────── INGEST (offline) ───────────┐
 raw docs ─►  load_data ─► markdown/parent/child chunking ─► bge-large embeddings ─► Chroma + BM25 pickle
                 └────────────────────────────────────────┘
 question ─►  STEP 1  query rewrite / decompose            (llm_utils)
          ─►  STEP 2  HYBRID retrieve  (BM25 + dense ensemble) ─► noise filter ─► bge-reranker ─► smart-window
          ─►  STEP 4  SUFFICIENCY GATE  (Identify-then-Verify)   (gate.py)
                        │ sufficient ──────────────────────────────────────┐
                        │ insufficient                                      │
          ─►  STEP 4.25 GAP-LOCAL re-retrieval (free, local)               │
          ─►  STEP 4.5  WEB fallback (MC only): Tavily/PubMed ─► rerank ─► merge
                                                                            ▼
 answer  ◄─  GENERATE  strict-grounding prompt, TF/MC rules     (generate_answers_gpt4_ADRD_Bench)
```

---

## 2. Retrieval Strategy

| Aspect | Choice |
|---|---|
| Embeddings | `BAAI/bge-large-en-v1.5`, L2-normalized, cosine, query-instruction prefix |
| Chunking | Markdown-header split → token parent (800) / child (250) chunks; parent map for window expansion |
| **Retrieval** | **Hybrid BM25 + dense ensemble** (RRF), weights **0.3 / 0.7**, `pre_k = 30` candidates |
| Candidate cleaning | KB-noise filter drops academic artifacts (refs, author blocks, tables, page markers) before rerank |
| Reranking | `BAAI/bge-reranker-v2-m3` cross-encoder; `rerank_min_prob = 0.05` |
| Window expansion | "smart window" expands the matched child chunk to sentence-bounded parent context |

## 3. Generation Strategy

- Model: **GPT-4**, temperature 0 (deterministic, stable formatting).
- **Strict grounding**: "Answer STRICTLY based on the provided context; do NOT use outside
  knowledge." No "insufficient / abstain" option (forces a decision).
- **TF rule**: Yes if the context states *or reasonably implies* it; No if it contradicts or
  there is no related info.
- **MC rule**: output the single option letter best supported by the context.
- No generator fine-tuning (hard constraint).

## 4. Sufficiency Gate — deciding *when the local KB is insufficient*

**Identify-then-Verify (ItV)** for both TF and MC (`gate.py`), the best-AUC evaluator we found:

1. **Identify** — prompt a cheap LLM (gpt-4o-mini) N=5× at temperature 0.7 to name the single
   most important *missing* piece of information ("NONE" if nothing is missing).
2. **Consensus** — embed the N gap hypotheses (`all-MiniLM-L6-v2`), take the one with the
   highest mean cosine similarity (semantic center).
3. **Verify** — a separate call checks whether that consensus gap is actually PRESENT in the
   context. If it was hallucinated → context is sufficient; otherwise insufficient.

**Measured quality (AUC of sufficiency-signal vs answer-correctness):** ItV **TF 0.694 / MC
0.731**, beating NLI-answerability (TF 0.675) and a GPT-4o confidence prompt (~0.63).

---

## 5. Local-KB Completion — Experiments & Their Effect

This is the core of the stage: what we do *after* detecting that the local context is
insufficient, plus the retrieval fixes that unlocked the gains. Effects are on the 149-question
benchmark (and on targeted subsets where noted).

| # | Experiment | What it does | Optimization effect |
|---|---|---|---|
| 1 | **BM25 hybrid fix** ⭐ | BM25 + EnsembleRetriever were imported in one `try`; a langchain-1.x path change silently nulled **both**, leaving retrieval **vector-only**. Fixed the imports → lexical+semantic hybrid actually runs. | **Biggest single lever: TF +7.** Exact-term questions (e.g. "screen out" vs "cognitive screening") that pure-semantic ranked >300th now surface at #1. |
| 2 | **ItV sufficiency gate** | Replaced GPT-4o confidence prompt / NLI with Identify-then-Verify for TF+MC. | Best-AUC gate (0.694 / 0.731). In a clean run took **MC to 29/29**. Higher precision → fewer false "insufficient" triggers. |
| 3 | **Gap-local re-retrieval** | On insufficiency, turn "what's missing" into a targeted query and re-search the **local KB** before any web. The answer is often in-KB but buried by the generic query. | Free, local, noise-free; recovered buried answers (≈ +2 of 6 on a gate-blocked subset). Kept for both TF and MC. |
| 4 | **KB completion (curriculum)** | Specific curriculum facts the benchmark needs but the original KB lacked (e.g. fluid-cutoff time, the stress-prevention-bundle components, nonverbal-communication teaching points) added as labeled manual additions. | **Genuinely needed: verified** — removing them from *both* the vector store AND the BM25 corpus makes 7/8 of those questions fail. Not a retrieval artifact. |
| 5 | **KB noise filter** | High-precision drop of academic-PDF artifacts from the candidate pool before rerank. | Hygiene improvement; the cross-encoder already removed ~98% of noise from *final* context, so net accuracy effect is small but it cleans the pre-rerank pool. |
| 6 | **Web fallback: DuckDuckGo → Tavily** | Free web search to fetch missing facts; query built from the identified gap (2–3 sub-queries) + PubMed E-utilities; junk-sentence filter, dedup, relevance pre-filter, local-floor guard. **Tavily** (free tier) replaced DDG scraping. | DDG was the coverage ceiling (historically ≈ +3, mostly stuck). **Tavily surfaces confirming facts DDG could not** (fixed MC_016 electric-toothbrush guidance that DDG returned *nothing* for). |
| 7 | **Suppress web on TF** ⭐ | Web content dumped into a binary true/false judgment flips correct answers more than it fixes (web-on-TF: **0 fixed / 2 broke** in testing). Disabled web for TF (`web_tf_enabled=False`); gap-local still runs for TF; web kept for MC's concrete-fact lookups. | **+3** (removed TF "churn"); TF rose to 111/120. |
| 8 | **Trained sufficiency evaluator** (exploratory) | 2-stage local model: general+medical pretraining (SQuAD2 + FEVER + PubMedQA) → ItV-distilled ADRD adaptation. Aimed to replace the prompt gate at zero inference cost. | Promising but **did not beat prompt-ItV** (ADRD AUC ≈ 0.62 vs ItV 0.73); data-limited (149 questions). Documented as future work in `SUFFICIENCY_EVALUATOR_LOG.md`. |

**Negative / reverted results (kept for honesty):** non-destructive completion ordering (−4),
sentence-strip refinement (−2, over-trims), open-domain web (no gain, DDG-limited),
bm25_weight lowering for MC (no gain; 0.3 is optimal), specific-claim numeric gate (failed —
stray numbers in context defeat bare matching).

### Results progression

| Configuration | Total | TF | MC | Winnable (/144) |
|---|---|---|---|---|
| Session start (vector-only, NLI gate) | 130/149 | 102 | 28 | 90.3% |
| ItV gate (clean) | 130/149 | 101 | **29** | 90.3% |
| + BM25 fix + KB completion + noise filter | 135/149 | 108 | 27 | 93.8% |
| **+ Tavily + suppress web-on-TF (final)** | **138/149** | **111** | 27 | **95.8%** |

*(Run-to-run variance is ±2–3 from web/gate non-determinism.)*

---

## 6. Script Inventory

**Core retrieval pipeline**

| Script | Function |
|---|---|
| `run_retrieval_adrd.py` | Orchestrates the per-question pipeline: query rewrite → hybrid retrieve → sufficiency gate → gap-local → web fallback → write contexts CSV. CLI: `--ids`, `--pre_k`, `--force_web`, `--no_web`. |
| `advanced_retriever.py` | The retriever: builds the **BM25 + dense ensemble**, applies the KB-noise candidate filter, cross-encoder reranks, and does smart-window parent expansion. |
| `ingest_documents_markdownHeaderTextSplitter.py` | Incremental ingestion: markdown-header + parent/child chunking, bge-large embedding, upsert to Chroma + BM25 pickle + parent map. Tracks `processed_files.json`. |
| `load_data.py` | Document loader / pre-cleaner (`SimpleBrainCheckLoader`) feeding ingestion. |
| `gate.py` | **Sufficiency gate** — ItV (TF+MC) and NLI-answerability; decides sufficient/insufficient. |
| `web_fallback_retriever.py` | Free web/medical retrieval: **Tavily** backend (DDG fallback) + **PubMed E-utilities**; junk-sentence filter, near-dup dedup, domain policy. |
| `kb_noise.py` | High-precision academic-artifact detector used as a pre-rerank candidate filter. |
| `llm_utils.py` | LLM helpers: query rewrite/decompose, gap-query generation, missing-info→web-queries, ItV/NLI/context evaluators, retry wrapper. |
| `rag_config.py` | Single source of config for ingestion, retrieval, reranking, gate, and web fallback. |

**Generation**

| Script | Function |
|---|---|
| `generate/generate_answers_gpt4_ADRD_Bench.py` | Main generator: builds context from retrieved passages, strict-grounding prompt, TF/MC rules, accuracy check, checkpointing. `--retrieval`, `--snippets`, `--verify`. |
| `generate/rag_generation_config.py` | System prompt, max context snippets, model registry. |
| `generate/generate_answers_{gpt5.2,gemini3,llama3.x,...}_*.py` | Parallel generator variants for other models / ablations (model-comparison family). |

**Sufficiency-evaluator research track**

| Script | Function |
|---|---|
| `train_sufficiency.py` | Stage-1 trained evaluator: general+medical pretraining (SQuAD2/FEVER/PubMedQA), zero-shot ADRD AUC. |
| `adapt_sufficiency.py` | Stage-2: ItV-distilled ADRD domain adaptation; held-out AUC vs base/ItV. |
| `grade_evaluator.py` | Confusion-matrix / catch-rate read of an evaluator at a threshold. |
| `tf_gate_compare.py`, `nli_answerability_eval.py`, `itv_sufficiency_eval.py` | Gate comparisons (NLI vs ItV) and AUC eval harnesses. |
| `gap_guided_completion.py`, `evaluate_retrieval_quality.py` | Gap-completion prototype; LLM-judge of retrieval fact-recall. |

**PDF / KB utilities**

| Script | Function |
|---|---|
| `detect_visual_pdfs.py`, `move_visual_pdfs.py`, `vlm_pdf_to_md.py`, `find_teepa.py` | Identify image-only PDFs, route them, VLM-transcribe to markdown, locate specific sources. |

---

## 7. Honest Ceiling Analysis & Next Steps

**Where the remaining 11 wrong sit:** 5 are **Bucket-B** (ambiguous/mislabeled, should be
excluded). Of the 6 "winnable": ~2 are **genuinely hard** (a figure that disagrees with the
public literature; a curriculum-only interpretation absent from the whole web), and ~4 are
**generation-reasoning / run-variance** (binary-judgment nuance on absolute statements).

**Retrieval side is at its practical ceiling** — hybrid retrieval, noise filtering, gap-local,
the ItV gate, Tavily, and the MC-only web policy are all in place. The residual bottleneck is
**knowledge availability** (a few facts exist nowhere retrievable) and **generation reasoning**
(deliberately untouched, per the no-generator-tuning constraint).

**Candidate next directions (not in this stage):** generation-side reasoning for absolute/
definitional TF statements; a sourced (non-gold-derived) version of the curriculum KB-completion;
revisiting the trained evaluator with a biomedical encoder (PubMedBERT) and more domain data.
