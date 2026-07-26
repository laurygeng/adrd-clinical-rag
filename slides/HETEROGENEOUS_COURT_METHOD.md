# The Heterogeneous Verification Court
### A Deterministic Evidence-Admissibility Gate for Churn-Free Knowledge Injection in Clinical RAG

*Refinement note for review — definitions, equations, and empirical tables. Scope: the single
core contribution. The earlier "adversarial (pro/con) retrieval" component has been removed
(non-original and no measurable gain); evidence is now obtained by a plain targeted retrieval.*

---

## 1. Problem: Knowledge-Injection Churn

**Setup.** For a clinical True/False claim $q$, a **frozen** generator $\mathcal{G}$ produces a
local-only baseline answer
$$a_0 = \mathcal{G}(q, C_{\text{local}}) \in \{\text{Yes}, \text{No}\}.$$
When the local context $C_{\text{local}}$ is judged insufficient, we retrieve external evidence
$E$ and must decide whether $E$ should be allowed to **override** $a_0$.

**The failure mode.** Let $y$ be the gold label and $a_{\text{sup}}$ the answer after *naively*
appending $E$ to the context. We define the **churn event**
$$\textsf{Churn}(q) \;\triangleq\; \mathbb{1}\!\left[\,a_0 = y \;\wedge\; a_{\text{sup}} \neq y\,\right],$$
i.e. injected evidence flips a *previously correct* judgment. Empirically, naive supplementation
on our 120 TF questions nets **−4 to −5** (churn outweighs genuine fixes), making external
knowledge effectively unusable.

**Goal.** A gate $G(q, E)$ that admits $E$ to override $a_0$ **only** when it is safe, minimizing
churn while preserving genuine fixes $\mathbb{1}[a_0 \neq y \wedge a^\ast = y]$.

---

## 2. Overview

$G$ is a conservative conjunction of three specialist checks over the (claim, evidence) pair,
combined by a veto arbiter. Unlike homogeneous self-consistency voting (which lowers *random*
noise but shares *systematic* blind spots), the checks are **heterogeneous** — each targets a
distinct, empirically observed failure mode:

| Check | Failure mode blocked | Nature |
|---|---|---|
| Ontological admissibility | super-class evidence supporting a sub-type claim | **deterministic** (MeSH) |
| Modal admissibility | claim asserting stronger force than the evidence | LLM operator extraction |
| Factual support | evidence does not establish the claim | 3-vote self-consistency |
| Veto arbiter | any red flag ⇒ do not override | logical AND |

---

## 3. Method

### 3.1 Concept Grounding
An entity extractor $\sigma(\cdot)$ returns the medical subject a statement makes its claim about
(preferring the specific subtype when a general category co-occurs). A linker
$\phi:\text{term}\rightarrow\mathcal{C}$ maps it to a MeSH concept:
$$c_q = \phi(\sigma(q)), \qquad c_e = \phi(\sigma(E)).$$
MeSH is a rooted hierarchy $(\mathcal{C}, \prec)$; each concept carries **tree numbers** (e.g.
*Alzheimer Disease* $=$ `C10.228.140.380.100`, *Dementia* $=$ `C10.228.140.380`). A strict
tree-number prefix is a strict is-a relation:
$$c \prec c' \;\iff\; \exists\, t\in T(c),\, t'\in T(c') :\; t' \text{ is a strict prefix of } t,$$
where $T(\cdot)$ is the concept's set of tree numbers.

### 3.2 Ontological Admissibility (deterministic, MeSH-grounded)

> **Definition 1 (Ontology-boundary leakage).** Evidence about $c_e$ *leaks* onto a claim about
> $c_q$ iff $c_e$ is a strict hypernym (super-class) of $c_q$:
> $$\textsf{leak}(c_e, c_q) \;=\; \mathbb{1}\!\left[\, c_q \prec c_e \,\right].$$

Intuition: a property established for a broad class (*dementia*) cannot be attributed to a
specific subtype (*Alzheimer's disease*) — the tremor common across dementias is largely due to
*other* subtypes. We adopt the **high-precision** form (only strict hypernymy triggers a veto);
sibling/incomparable pairs and unmapped concepts are **deferred** to the other judges, because at
extraction time "incomparable" is too often an artifact of relational or multi-entity claims to
block deterministically. This check is fully reproducible and requires no key (MeSH lookup + SPARQL).

### 3.3 Modal Admissibility

An operator extractor $\mu(\cdot)$ returns the strongest deontic/epistemic/quantificational force
of a statement on a family-wise strength order $(\mathcal{M}, \preceq_\mu)$:
$$\text{deontic: } \textsf{may} \prec \textsf{should} \prec \textsf{must}, \quad
  \text{quantificational: } \textsf{some} \prec \textsf{most} \prec \textsf{all}, \quad
  \text{epistemic: } \textsf{possible} \prec \textsf{certain}.$$
Let $m_q = \mu(q)$, $m_e = \mu(E)$.

> **Definition 2 (Modal/degree swap).** A swap occurs iff the claim asserts strictly stronger
> force than the evidence supports, within the same family:
> $$\textsf{swap}(q, E) \;=\; \mathbb{1}\!\left[\, m_q \succ_\mu m_e \,\right].$$

Example: claim "one *must* make them follow the rules" (obligation) vs. evidence "rules *help*
engagement" (benefit) ⇒ $m_q \succ_\mu m_e$ ⇒ swap. This is the natural-logic observation that
strengthening an operator in an upward-monotone context breaks entailment.

### 3.4 Factual Support (self-consistency)

The high-variance semantic judgment is stabilized by $K$-sample majority voting. With per-sample
verdict $f_k \in \{+1, -1, 0\}$ (TRUE / FALSE / NO-INFO), $k = 1,\dots,K$ (we use $K=3$):
$$\hat{f} \;=\; \operatorname{mode}\big(f_1, \dots, f_K\big).$$
Each sample judges by the *overall weight* of the literature (a single null-result study does not
override a clearly stated general principle) and applies a **medical-ontology rule** distinguishing
syndrome / disease / disorder levels.

### 3.5 Veto Arbiter

$$
G(q, E) =
\begin{cases}
\textsf{INSUFFICIENT}, & \textsf{leak}(c_e, c_q) \;\vee\; \textsf{swap}(q, E) \;\vee\; \hat{f} = 0 \\[4pt]
\hat{f}, & \text{otherwise}
\end{cases}
$$
$$
a^\ast =
\begin{cases}
a_0, & G(q,E) = \textsf{INSUFFICIENT} \\
\text{map}(\hat{f}), & \text{otherwise}
\end{cases}
\qquad \text{map}(+1)=\text{Yes},\; \text{map}(-1)=\text{No}.
$$
The gate overrides the baseline **only** when all three admissibility conditions clear — a
conservative logical AND that is the formal expression of "block churn first."

### Algorithm

```
Input:  claim q, baseline a0, retrieved evidence E
Output: final answer a*
1  c_q, c_e  <- phi(sigma(q)), phi(sigma(E))              # concept grounding (MeSH)
2  if leak(c_e, c_q):            return a0                 # Def 1  (deterministic veto)
3  if swap(q, E):                return a0                 # Def 2
4  f_hat <- mode(f_1..f_K)                                 # 3-vote factual support
5  if f_hat == NO_INFO:          return a0
6  return map(f_hat)                                       # admit override
```

---

## 4. Relation to Prior Work (novelty positioning)

The underlying logical principles are **not** claimed as new. Ontology-boundary and modal checks
are grounded in **natural logic / monotonicity** (MacCartney & Manning, 2009) and
**factuality/veridicality** (Saurí & Pustejovsky, 2009); the hierarchy comes from **MeSH/UMLS**
(Bodenreider, 2004). Self-consistency voting follows Wang et al. (2023).

**Our contribution is applied and system-level:** (i) we formalize *knowledge-injection churn* as
the dominant failure mode when a strong frozen generator is augmented with external evidence in a
clinical TF setting; (ii) we instantiate ontological + modal monotonicity as a **deterministic,
MeSH-grounded evidence-admissibility gate** that eliminates this churn under a frozen generator and
free APIs; (iii) the gate is derived from, and validated against, a case-level error analysis of a
real ADRD caregiving benchmark. It is a *minor but reproducible* mechanism, not a new theory.

---

## 5. Empirical Validation

**(a) Ontology gate — unit tests (deterministic, no LLM).** 5/5.

| claim $c_q$ | evidence $c_e$ | relation | $G$ | expected |
|---|---|---|---|---|
| Alzheimer Disease | Dementia | evidence broader | **LEAK** | LEAK |
| Dementia | Alzheimer Disease | evidence narrower | OK | OK |
| Delirium | Delirium | same | OK | OK |
| Alzheimer Disease | Parkinson Disease | incomparable | OK (defer) | defer |
| Dementia | Delirium | incomparable | OK (defer) | defer |

**(b) End-to-end (extract → MeSH gate) on diagnostic cases.**

| case | claim subj | evidence subj | gate | note |
|---|---|---|---|---|
| TF_046 (churn) | Alzheimer's disease | dementia | **LEAK** ✓ | blocks the super-class leakage |
| TF_023 (fix) | delirium | delirium | OK ✓ | preserves a genuine fix |
| delirium-in-dementia | delirium | dementia | OK ✓ | no false positive on a relational claim |
| vascular dementia | vascular dementia | dementia | UNKNOWN | linking gap → safe defer |

**(c) Court ablation on 120 TF (baseline = frozen GPT-4 + local, 109/120 correct).**

| system | net | note |
|---|---|---|
| naive supplementation | −4 to −5 | churn dominates |
| Court, no defenses | −4 | 4 breaks: {TF_063, TF_075, TF_099, TF_113} |
| Court + ontology rule + 3-vote | −2 | breaks: {TF_099, TF_113} |
| Court + **deterministic MeSH ontology gate** | **−1** | only break: {TF_113} |

Component effects: the **medical-ontology rule** fixed TF_063 (syndrome vs. disease vs. disorder);
**3-vote self-consistency** fixed TF_075 (variance-driven flip). The best configuration leaves a
single break, **TF_113**, where the system is *more correct than the gold* — it flagged "90% of
communication is nonverbal" as a debunked urban legend, retrieving the exact refuting source. The
**method-attributable regression is therefore $\approx 0$** (baseline drifts 109–110 across runs
under temperature-0 generation, so the −1/−2 difference is within run variance).

**On the deterministic MeSH gate specifically.** In isolation it is validated (Table (a), 5/5) and
integrated as a high-precision veto. On this benchmark, however, ontology-boundary leakage is a
**rare** failure mode — essentially TF_046 alone, which the modal/factual judges already keep
correct — so the gate seldom fires and its *aggregate* effect here is marginal. Its contribution is
qualitative: it replaces an LLM heuristic with a **deterministic, reproducible, citable** mechanism
for the cases where the failure mode does occur, and would carry more weight on corpora richer in
sub-type/super-class confusions.

---

## 6. Limitations

- **Entity-linking coverage.** Some surface variants (e.g. "vascular dementia") do not resolve to a
  MeSH descriptor and return UNKNOWN (safe defer, no over-blocking). A dedicated biomedical linker
  (scispaCy + UMLS) would raise coverage.
- **Scope of the ontology gate.** It handles single-entity subtype substitution; relational /
  multi-entity claims are deferred to the modal and factual judges.
- **Benchmark gold quality.** Legacy errors in the evaluation set (e.g. TF_113) cap the measurable
  score and are reported as a qualitative finding rather than chased.
