#!/usr/bin/env python3
"""
Sufficiency GATE: decides whether retrieved context is enough to answer.
  - TF (verification)  -> local NLI answerability  (validated AUC 0.675 > GPT-4o 0.630)
  - MC (preference)    -> Identify-then-Verify      (validated AUC 0.731 vs NLI 0.45)
Self-contained: NLI runs locally; ItV uses a cheap LLM (gpt-4o-mini) for identify/verify.
"""
import numpy as np
from rag_config import config
from llm_utils import get_openai_client, _chat_with_retry

_nli = None
_emb = None

def _get_nli():
    global _nli
    if _nli is None:
        from sentence_transformers import CrossEncoder
        _nli = CrossEncoder(getattr(config, "nli_model", "cross-encoder/nli-deberta-v3-base"))
    return _nli

def _get_emb():
    global _emb
    if _emb is None:
        from sentence_transformers import SentenceTransformer
        _emb = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb

# ---------------- TF: NLI answerability ----------------
def tf_sufficient(passages, statement):
    nli = _get_nli()
    id2 = {int(k): v.lower() for k, v in nli.model.config.id2label.items()}
    lab = {v: k for k, v in id2.items()}
    ENT, CON = lab["entailment"], lab["contradiction"]
    ps = [p for p in passages[:12] if p]
    if not ps:
        return False, 0.0
    pr = np.asarray(nli.predict([(p, statement) for p in ps], apply_softmax=True, show_progress_bar=False))
    answerability = float(max(pr[:, ENT].max(), pr[:, CON].max()))  # decisively entailed OR contradicted
    return answerability >= getattr(config, "nli_suff_threshold", 0.5), answerability

# ---------------- MC: Identify-then-Verify ----------------
def _is_none(s):
    s = (s or "").strip().upper()
    return s.startswith("NONE") or s == ""

def mc_sufficient(question_block, context, n=None):
    n = n or getattr(config, "itv_n", 5)
    client = get_openai_client()
    model = getattr(config, "llm_gap_model", "gpt-4o-mini")
    sysp = ("You judge whether a CONTEXT is sufficient to answer a multiple-choice question. "
            "Name the SINGLE most important piece of information still MISSING from the context needed "
            "to confidently determine the correct option. If the context already contains everything "
            "needed, answer exactly 'NONE'. Answer with one short phrase only.")
    gaps = []
    for _ in range(n):
        r = _chat_with_retry(client, model=model, temperature=0.7, max_tokens=40,
            messages=[{"role": "system", "content": sysp},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question_block}"}])
        gaps.append((r.choices[0].message.content or "").strip())
    none_frac = float(np.mean([_is_none(g) for g in gaps]))
    if none_frac >= 0.6:
        return True, none_frac
    real = [g for g in gaps if not _is_none(g)]
    if not real:
        return True, none_frac
    E = _get_emb().encode(real, normalize_embeddings=True)
    consensus = real[int((E @ E.T).mean(1).argmax())]
    v = _chat_with_retry(client, model=model, temperature=0, max_tokens=4,
        messages=[{"role": "system", "content": "Decide if a specific piece of information is PRESENT in the context. Answer exactly 'PRESENT' or 'ABSENT'."},
                  {"role": "user", "content": f"Context:\n{context}\n\nInformation: {consensus}\n\nIs this present in the context?"}])
    present = "PRESENT" in (v.choices[0].message.content or "").upper()
    return present, none_frac   # present => the claimed gap was hallucinated => sufficient

# ---------------- dispatch ----------------
def is_sufficient(q_type, question, passages):
    """question already includes the options block for MC (as built in run_retrieval)."""
    try:
        if not passages:
            return False
        if q_type == "TF":
            return tf_sufficient(passages, question)[0]
        return mc_sufficient(question, "\n\n".join(passages[:8]))[0]
    except Exception:
        return False
