# Reading Notes — the "improve" direction (churn-free, generalizable, free-text ready)

Three core papers, each with: what it says · relation to our idea · what we borrow · our delta.
The through-line: **an answer must not exceed its evidence** — decide *when* to inject (Mallen),
*whether a passage is admissible* (Yoran, via NLI), and *verify each atomic claim* (FActScore).

---

## 1. Mallen et al. 2023 — *When Not to Trust Language Models* (arXiv 2212.10511, ACL 2023)

**What it says.** LMs reliably know **popular** entities but fail on the **long tail**, and scaling
does not fix the tail. Retrieval augmentation helps *a lot* on low-popularity facts but is
**unnecessary — and can add cost/noise — on popular facts the model already knows**. They propose
**adaptive retrieval**: retrieve only when the entity popularity is below a threshold, else use
parametric memory. Introduces **PopQA** (14k long-tail entity questions).

**Relation to us.** This is the **empirical foundation of our "churn"**: external knowledge helps
on genuine gaps but hurts/wastes when the baseline is already right. Their *adaptive retrieval* is
a **gating decision at the trigger level** — our admissibility gate is the same idea, one step
later (at the evidence-override level).

**What we borrow.** (a) The framing "inject only when the baseline is likely insufficient"; (b) the
methodology of plotting help-vs-hurt as a function of a **signal**; (c) **PopQA as a candidate 2nd
dataset** — long-tail, injection is genuinely needed, so churn is visible.

**Our delta.** Their signal is **entity popularity** (needs entity metadata, entity-centric QA
only). Ours is **evidence–claim entailment + baseline confidence**, which is domain-agnostic and
**works for free-text**, not just entity lookups. Frozen generator (they don't fine-tune either).

---

## 2. Yoran et al. 2023 — *Making RALMs Robust to Irrelevant Context* (arXiv 2310.01558)

**What it says.** States the desideratum "**relevant context helps, irrelevant context does not
harm**" — and shows it is **often violated** (irrelevant passages degrade accuracy, especially in
multi-hop). Two methods: (1) **NLI-based filtering** — keep a passage only if an NLI model confirms
it entails the question–answer pair (prevents degradation but discards some relevant passages as
"collateral damage"); (2) **robustness training** — fine-tune on mixed relevant/irrelevant contexts
(**~1,000 examples suffice**). Evaluated on 5 open-domain QA benchmarks.

**Relation to us.** Their desideratum **is our problem statement almost verbatim** (= churn-free
injection). Their **NLI filtering is exactly the domain-agnostic mechanism we want** — it
**replaces the MeSH ontology with a general entailment check**, directly answering the advisor's
"you don't need an ontology, define your own method."

**What we borrow.** (a) **NLI as the general, no-ontology admissibility mechanism**; (b) their
desideratum as our formal problem framing; (c) the "collateral damage" precision/recall tradeoff —
our **high-precision** gate is a principled response to it.

**Our delta.** Their NLI checks *relevance* ("does the passage entail the Q–A?"). We add the
**direction**: evidence must entail the claim **at its own specificity and strength**, not a
**broader or weaker** version — that is our contribution *over* plain NLI filtering (subsumes both
the "ontology-boundary" and "modal-swap" cases into one entailment principle). Also **we do not
fine-tune** (they do); we keep the generator frozen.

---

## 3. Min et al. 2023 — *FActScore* (arXiv 2305.14251, EMNLP 2023)

**What it says.** For **long-form generation**, decompose the output into **atomic facts**
(individual verifiable propositions), verify each against a knowledge source (retrieval + LM), and
report **% supported**. An **automated estimator** matches humans within **<2% error**; e.g.
ChatGPT scores **58%** on biography generation — "generations are a mixture of supported and
unsupported pieces," so binary judgments are insufficient.

**Relation to us.** This is our **vehicle for the free-text generalization**. For a free-text ADRD
answer, decompose into atomic claims and apply our admissibility gate **per claim**. It also gives
us a **free-text evaluation metric**, so we are not restricted to TF/MC accuracy.

**What we borrow.** (a) The **atomic decomposition + per-claim verification** pipeline as the
free-text instantiation of "evidence-bounded answering"; (b) the **automated estimator**
methodology; (c) FActScore itself as an evaluation metric for the generative dataset.

**Our delta.** FActScore checks **support (binary)**. We add the **specificity/strength dimension**:
a claim can be loosely "supported" yet **over-specific or over-strong** relative to the evidence —
detecting that over-claiming is our contribution.

---

## Synthesis — where our contribution sits

| paper | mechanism | granularity | our extension |
|---|---|---|---|
| Mallen | when to inject (popularity signal) | question | general signal (entailment+confidence), free-text |
| Yoran | NLI relevance filter | passage | + specificity/strength **direction**; no fine-tuning |
| FActScore | support check | atomic claim | + over-claiming (over-specific / over-strong) |

**One-line positioning:** *a **specificity/strength-preserving admissibility** principle —
evidence may support an answer only if it entails the claim at the claim's own specificity and
strength — instantiated with a general entailment check (no ontology), applied at claim level so it
covers TF, MC, and free-text, under a frozen generator.*

**Reading order:** Mallen (why injection hurts) → Yoran (general NLI mechanism, no ontology) →
FActScore (free-text per-claim verification).
