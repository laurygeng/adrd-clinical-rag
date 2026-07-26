#!/usr/bin/env python3
"""
trace_logger.py
Centralized run-scoped logging utilities.

Creates:
  logs/<RUN_TS>/critic/*.md
  logs/<RUN_TS>/critic/critic_traces.jsonl
  logs/<RUN_TS>/orchestrator/pipeline_traces.jsonl
"""

import os
import json
import threading
from datetime import datetime
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_RUN_TS: Optional[str] = None
_RUN_DIR: Optional[str] = None

def get_project_root() -> str:
    # core/trace_logger.py -> project_root/core/trace_logger.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_run_ts() -> str:
    global _RUN_TS
    if _RUN_TS is None:
        # Keep stable for whole process
        _RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _RUN_TS

def get_run_dir() -> str:
    global _RUN_DIR
    if _RUN_DIR is None:
        root = get_project_root()
        _RUN_DIR = os.path.join(root, "logs", get_run_ts())
        os.makedirs(_RUN_DIR, exist_ok=True)
    return _RUN_DIR

def ensure_subdir(name: str) -> str:
    d = os.path.join(get_run_dir(), name)
    os.makedirs(d, exist_ok=True)
    return d

def _safe_filename(s: str, max_len: int = 120) -> str:
    # make filesystem-friendly
    keep = []
    for ch in (s or ""):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    if len(out) > max_len:
        out = out[:max_len]
    return out or "item"

def make_item_id(question_id: Optional[str], question: str) -> str:
    if question_id:
        return _safe_filename(question_id)
    # fallback: short hash prefix
    import hashlib
    h = hashlib.md5((question or "").encode("utf-8")).hexdigest()[:10]
    return f"Q_{h}"

def write_jsonl(subdir: str, filename: str, obj: Dict[str, Any]) -> str:
    d = ensure_subdir(subdir)
    path = os.path.join(d, filename)
    line = json.dumps(obj, ensure_ascii=False)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return path

def write_text(subdir: str, filename: str, text: str) -> str:
    d = ensure_subdir(subdir)
    path = os.path.join(d, filename)
    with _LOCK:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return path

def write_run_meta(meta: Dict[str, Any]) -> str:
    path = os.path.join(get_run_dir(), "run_meta.json")
    with _LOCK:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
    return path
