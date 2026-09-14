"""
Compatibility shim for previous `local_google_shim.py` users.
Re-exports `Client` from `core.modelB`.
"""

from core.modelB import Client

__all__ = ["Client"]