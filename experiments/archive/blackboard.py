#!/usr/bin/env python3
"""Shared Blackboard — the working-memory object all agents read/write.

A plain dataclass (deterministic, transparent, trivially serializable for ablation logging).
Every field is written by exactly one agent and read by downstream agents, so the control flow
and each agent's contribution stay 100% inspectable.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Blackboard:
    question: str
    q_type: str                                   # "TF" | "MC"
    gold: Any = None                              # for eval only
    correct_letter: Any = None

    queries: List[str] = field(default_factory=list)          # Query Agent
    baseline_answer: Optional[str] = None                     # Answer Agent (local-only draft)
    baseline_confidence: Optional[float] = None               # self-consistency of the draft

    retrieved_evidence: Dict[str, List[str]] = field(default_factory=lambda: {"local": [], "web_pro": [], "web_con": []})

    sufficiency: Optional[str] = None            # Judge Panel: "sufficient"|"insufficient"
    gap: Optional[str] = None                    # Judge Panel: what's missing

    judge_flags: Dict[str, Any] = field(default_factory=lambda: {"entity_match": None, "modal_match": None, "fact": None})

    final_decision: Optional[str] = None         # Answer Agent
    trace: List[str] = field(default_factory=list)

    def log(self, msg: str):
        self.trace.append(msg)
