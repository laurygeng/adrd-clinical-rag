#!/usr/bin/env python3
"""Multi-agent Verify/Debate Panel (agent ⑤) for TF questions.

For a True/False statement the single-pass generator can mishandle absolute/definitional
nuance ("never", "always", "X IS Y"). This panel runs a structured debate around the FROZEN
generator (no fine-tuning): a Proponent argues the statement is True, an Opponent argues it is
False (both grounded in the context), then a Judge adjudicates on the context + both arguments.

Debaters use a cheap model (gpt-4o-mini); the Judge uses the same model as the answer agent
(gpt-4) so the A/B isolates the *debate structure*, not the model.
"""
import os
from openai import OpenAI

_client = None
def _c():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

DEBATER_MODEL = "gpt-4o-mini"
JUDGE_MODEL   = "gpt-4"

def _chat(model, sys, user, max_tokens=180, temp=0.3):
    r = _c().chat.completions.create(model=model, temperature=temp, max_tokens=max_tokens,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()

def _argue(statement, context, side):
    verdict = "TRUE" if side == "pro" else "FALSE"
    sys = (f"You are the {'PROPONENT' if side=='pro' else 'OPPONENT'} in a debate about a True/False "
           f"statement in dementia caregiving. Using ONLY the provided context, make the strongest "
           f"concise case (2-4 sentences) that the statement is {verdict}. Cite specific context. "
           f"If the context genuinely does not support {verdict}, say so honestly.")
    return _chat(DEBATER_MODEL, sys, f"Context:\n{context}\n\nStatement: {statement}\n\nArgue it is {verdict}:")

def debate_tf(statement, context):
    """Returns (verdict 'Yes'/'No', pro_arg, opp_arg)."""
    pro = _argue(statement, context, "pro")
    opp = _argue(statement, context, "opp")
    judge_sys = ("You are the JUDGE of a True/False statement in dementia caregiving. Decide based "
                 "STRICTLY on the context: answer 'Yes' if the context states OR reasonably implies "
                 "the statement is true; answer 'No' if the context contradicts it, or if a specific "
                 "claim it makes is not supported. Weigh both arguments but rely on the context. "
                 "Respond with EXACTLY one word: Yes or No.")
    judge_user = (f"Context:\n{context}\n\nStatement: {statement}\n\n"
                  f"Proponent (argues True):\n{pro}\n\nOpponent (argues False):\n{opp}\n\n"
                  f"Your verdict (Yes or No):")
    verdict = _chat(JUDGE_MODEL, judge_sys, judge_user, max_tokens=4, temp=0)
    v = "Yes" if verdict.strip().lower().startswith("y") else "No"
    return v, pro, opp
