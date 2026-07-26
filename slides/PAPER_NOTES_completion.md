# Literature Map — the COMPLETION step
### "After a sufficiency evaluator flags missing information, how do you supplement the correct knowledge without noise?"

This is our innovation angle: not gap *detection* (ItV already does that), but the **completion /
supplementation** step that follows. The completion step has three sub-problems; every paper below
addresses one or two, and the gap they leave open is our opportunity.

**Sub-problems of completion:**
- **(T) Targeting** — fetch the *right* missing knowledge, not just more text.
- **(D) Denoising** — don't drag in irrelevant/wrong content while supplementing.
- **(S) Stopping / iteration** — know what is still missing and when to stop.
- **(C) Churn-safety** — *(under-addressed)* supplement to fill a gap without disturbing what the
  baseline already had right (the failure mode we measured).

---

## Cluster 1 — Missing-information-guided targeted retrieval  *(T, most on-point)*

**MIGRES** — *Missing-Information Guided Retrieve-Extraction-Solving* (arXiv 2404.14043).
The closest paper to our angle. It verifies that LLMs can both extract information **and name what
is missing**, then uses the identified gap to **generate a targeted query** that steers the next
retrieval; a **sentence-level re-ranking filter** strips irrelevant content before extraction; the
loop repeats until solved. → Gives us both **(T)** targeted, gap-driven querying and **(D)** a
sentence-level filter. Tested on multi-hop QA.

**Self-Ask** (Press et al. 2022, arXiv 2210.03350). Decomposes a question into explicit **follow-up
sub-questions**, answers each (optionally via search). The unanswered sub-question *is* the
identified gap. → **(T)** via decomposition.

**IRCoT** (Trivedi et al. 2022, arXiv 2212.10509). **Interleaves retrieval with chain-of-thought**:
each reasoning step guides the next retrieval, so knowledge is fetched exactly where the reasoning
needs it. → **(T)+(S)** reasoning-guided, incremental filling.

---

## Cluster 2 — Corrective / confidence-triggered supplementation  *(T + D + trigger)*

**CRAG** — *Corrective RAG* (arXiv 2401.15884). A lightweight **retrieval evaluator** labels the
retrieval *correct / incorrect / ambiguous*; if not correct, it triggers **web-search
supplementation** plus a **decompose-then-recompose knowledge-refinement** that strips a document to
strips/knowledge-strips and keeps only the relevant ones. → detect-insufficiency → supplement +
**(D)** refine. Directly parallels our pipeline.

**FLARE** — *Active RAG* (Jiang et al. 2023, arXiv 2305.06983). While generating, if the next
sentence is **low-confidence**, it forms a query from that sentence and retrieves to fill it before
continuing. → **confidence-triggered** supplementation (a signal-driven version of "when to
complete"). Relevant to free-text generation.

---

## Cluster 3 — Iterative retrieve↔generate loops  *(S)*

**ITER-RETGEN** (Shao et al. 2023, arXiv 2305.15294). The **model's own output reveals what is
still needed**, which informs the next retrieval, which improves the next output — iterate. Processes
retrieved knowledge holistically. → **(S)** the generation itself is the "what's still missing"
signal. Multi-hop QA, fact verification, commonsense.

---

## Cluster 4 — Denoising / refining the supplemented content  *(D, the key to "without noise")*

**RECOMP** (Xu et al. 2023, arXiv 2310.04408). **Compresses retrieved documents into concise
summaries before use** — an **extractive** compressor (pick useful sentences) and an **abstractive**
one (synthesize across docs); crucially it can **return an empty string when nothing is relevant**
(selective augmentation), reaching ~6% of original length with minimal performance loss. → the
cleanest **(D)** mechanism: only the distilled, relevant knowledge reaches the generator.

**CRAG knowledge refinement** (above) — decompose-then-recompose is the same idea at passage level.

---

## Cluster 5 — Self-critique of the supplemented knowledge  *(D + correctness)*

**Self-RAG** (Asai et al. 2023, arXiv 2310.11511). Trains reflection tokens: **IsRel** (is the
retrieved passage relevant), **IsSup** (does it support the statement), **IsUse** (is it useful) —
the model critiques each retrieved passage and its own output. → per-passage **admissibility**, but
requires fine-tuning.

**Chain-of-Verification** (Dhuliawala et al. 2023, arXiv 2309.11495). Generates verification
questions to check its own answer before finalizing.

**CaLM** (arXiv 2403.06857). Contrasts a large and a small model to verify grounded generation.

---

## What the field leaves open → our opportunity

1. **Almost everything above assumes "more relevant knowledge ⇒ better."** The methods target (T),
   denoise (D), and iterate (S) — but they are built for **multi-hop factoid QA where you must
   gather facts**. None addresses the regime we measured: a **strong baseline that is already mostly
   right**, where supplementing *relevant, even correct* evidence can still **flip a correct binary/
   decisive answer** (churn). Sub-problem **(C) churn-safety is essentially unaddressed.**
2. **Denoising (RECOMP, CRAG) reduces irrelevant text, but does not check whether admitting the
   evidence would over-turn a confident baseline** — it is relevance-based, not decision-risk-based.
3. **Self-RAG's admissibility needs fine-tuning;** we are constrained to a **frozen generator + free
   APIs**.

**Our slot (the completion-step innovation):**
> a **targeted, verified, *minimal-and-conservative* completion**: fill the *specific* identified
> gap (like MIGRES) with **distilled** content (like RECOMP), but gate admission by
> **decision-risk / entailment-direction** (our churn-safety check) rather than mere relevance, and
> do it **without fine-tuning**. In binary/MC tasks this shows as *fill-without-churn*; in free-text
> it shows as *completeness gain without contradiction*.

**Positioning vs the closest work:** MIGRES = targeted + sentence-filter; CRAG = corrective +
refine; RECOMP = compress-or-empty; Self-RAG = per-passage critique (fine-tuned). **Our delta =
completion is gated by whether admitting it would overturn a confident baseline (churn-safety),
frozen generator, and evaluated for free-text completeness — a dimension none of them optimize.**

---

## Suggested reading order
MIGRES (our exact shape: identify-missing → targeted retrieve → filter) → CRAG (corrective +
refine) → RECOMP (denoise / compress-or-empty) → Self-RAG (per-passage admissibility) → FLARE /
ITER-RETGEN (when-to-complete signals for free-text).
