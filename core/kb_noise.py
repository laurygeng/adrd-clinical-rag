#!/usr/bin/env python3
"""High-precision academic-artifact noise detector for the KB.

The knowledge base was built partly from research PDFs, which contributed non-content
fragments (author/affiliation blocks, reference lists, table/figure numeric dumps, page
markers, methods boilerplate). These crowd the pre-rerank candidate pool with medium-score
junk and starve genuine content — so the cross-encoder never gets to see the answer chunk.

Used as a CANDIDATE filter (drop noise from the pre_k pool BEFORE reranking), it is
non-destructive (the vector DB is untouched) and reversible via config.kb_noise_filter.

Design priority is PRECISION: Q&A items ('Question:'/'Answer:') are always protected, and
every rule has a word-count guard so real prose that merely mentions a study is not dropped.
"""
import re

AUTH    = re.compile(r'\b(PhD|MD|MSc|MPH|MBBS|FRCP|DPhil|Professor)\b')
ORG     = re.compile(r'\b(University|Division|Department|Institute|College|Hospital|Centre|Center|Faculty)\b')
CITE    = re.compile(r'[A-Z][a-z]+\s+\d{4},\s*\d+,\s*\d+|\bdoi:|\bISSN\b|\bet al\.,?\s*\d|\bpp\.\s*\d+|\bVol\.\s*\d+', re.I)
REFLIST = re.compile(r'^\s*\d+\.\s+[A-Z][a-z]+\s+[A-Z]{1,2}[,.]')          # "13. Guo J, Wang J,"
TBL     = re.compile(r'\b(Table|Figure|Fig\.)\s*\d', re.I)
METH    = re.compile(r'(search strateg|data sources and|inclusion criteri|exclusion criteri|PRISMA|databases were searched|systematic review and meta-analysis)', re.I)
PAGE    = re.compile(r'^\W*\d+\s+of\s+\d+\b|^[\d\s.,;:()%\-–—●•]+$')

def _strip(d):
    return re.sub(r'^Content:\s*\[[^\]]*\]\s*', '', re.sub(r'^Content:\s*', '', d)).strip()

def is_noise(text):
    t = _strip(text)
    if t[:30].lower().startswith(("question:", "answer:")):   # always protect Q&A
        return False
    nw = len(re.findall(r"[A-Za-z']+", t))
    if nw < 6:                       return True
    if PAGE.match(t):                return True
    if AUTH.search(t) and ORG.search(t): return True
    if REFLIST.match(t):             return True
    if TBL.search(t)  and nw < 40:   return True
    if CITE.search(t) and nw < 70:   return True
    if METH.search(t) and nw < 80:   return True
    return False
