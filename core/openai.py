"""
Compatibility shim for previous `openai.py` users.
Re-exports `OpenAI` from `core.modelA`.
"""

from core.modelA import OpenAI

__all__ = ["OpenAI"]