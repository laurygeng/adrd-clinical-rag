# core/integration_agent.py
"""
Integration Agent (Context Orchestrator)
Role: Validates, deduplicates, and strategically assembles heterogeneous information
(Base, Gap, Web) into a final cohesive context window based on task constraints.
"""

import hashlib
from typing import List, Dict, Any, Union

def _deduplicate_passages(passages: List[str]) -> List[str]:
    seen_hashes = set()
    unique: List[str] = []
    for p in passages or []:
        if not p: continue
        h = hashlib.md5(p.strip()[:100].encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(p)
    return unique

def _format_evidence_items(evidence_items: List[Union[str, Dict[str, Any]]]) -> List[str]:
    out: List[str] = []
    for ev in evidence_items or []:
        if isinstance(ev, str):
            t = ev.strip()
            if t: out.append(t)
            continue
        
        source = (ev.get("source") or "").strip()
        title = (ev.get("title") or "").strip()
        url = (ev.get("url") or "").strip()
        text = (ev.get("text") or "").strip()
        if not text: continue

        header = f"[WEB:{source}]"
        if title: header += f" {title}"
        if url: header += f" ({url})"
        out.append(header + "\n" + text)
    return out

def assemble_context(base_passages: List[str], gap_passages: List[str], web_evidence: List[Dict[str, Any]], q_type: str) -> str:
    """
    Strategically assemble context enforcing source limits and character budgets.
    """
    base_tagged = [f"[BASE] {p}" for p in (base_passages or [])[:4]]
    gap_tagged = [f"[GAP] {p}" for p in (gap_passages or [])[:3]]
    web_passages_trimmed = _format_evidence_items(web_evidence)[:1]

    is_qa = (q_type.strip().upper() == "QA")
    is_tf = (q_type.strip().upper() == "TF")

    # Priority routing logic based on question type
    if is_qa:
        prioritized_passages = base_tagged[:2] + web_passages_trimmed + gap_tagged + base_tagged[2:]
        MAX_CONTEXT_CHARS = 12000
        max_sources = 6
    else:
        prioritized_passages = gap_tagged + web_passages_trimmed + base_tagged
        MAX_CONTEXT_CHARS = 8000 if is_tf else 12000
        max_sources = 4 if is_tf else 6
        
    merged_passages = _deduplicate_passages(prioritized_passages)
    
    current_chars = 0
    final_passages = []
    
    # for p in merged_passages[:max_sources]:
    #     if current_chars + len(p) > MAX_CONTEXT_CHARS and len(final_passages) >= 2:
    #         break
    #     final_passages.append(p)
    #     current_chars += len(p)
        
    # return "\n\n".join(final_passages)


    for p in merged_passages[:max_sources]:
            # 把 >= 2 改为 >= 1，只要已经有一篇保底，后续超标就立刻熔断
            if current_chars + len(p) > MAX_CONTEXT_CHARS and len(final_passages) >= 1:
                break
            final_passages.append(p)
            current_chars += len(p)
            
    final_context = "\n\n".join(final_passages)
        
        # 物理级兜底：防止连第一篇文章自身就超过了 5000 字
    if len(final_context) > MAX_CONTEXT_CHARS:
            final_context = final_context[:MAX_CONTEXT_CHARS] + "\n...(Text truncated due to length limits)..."
            
    return final_context