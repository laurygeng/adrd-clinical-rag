#!/usr/bin/env python3
import os
import sys
import json
import math
import hashlib
import argparse
import pandas as pd
import nltk
from datetime import datetime
from tqdm import tqdm


def logit_to_prob(x):
    return 1 / (1 + math.exp(-max(min(x, 100), -100)))


def passage_id(src, text):
    """Stable dedup id (builtin hash() is process-salted and not reproducible)."""
    return f"{src}__{hashlib.md5(text.encode('utf-8')).hexdigest()}"


_CONF_RANK = {"low": 0, "medium": 1, "high": 2}

def conf_meets(conf, threshold):
    """True if a TF confidence level is >= the verify threshold (so it can SKIP web)."""
    return _CONF_RANK.get(str(conf).lower(), 0) >= _CONF_RANK.get(str(threshold).lower(), 2)


def run_eval(q_type, question, contexts):
    """Sufficiency via the NLI(TF)/ItV(MC) gate. Returns (satisfied, tf_verdict, tf_confidence, eval_missing, raw)."""
    suff = is_sufficient(q_type, question, contexts) if contexts else False
    return (suff, None, None, "", None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_config import config
from advanced_retriever import AdvancedRetriever
from llm_utils import (
    evaluate_context,
    rewrite_tf_query,
    evaluate_tf_evidence,
    decompose_mc_options,
    generate_web_queries_from_missing_info,
    generate_draft_query,
    agentic_search_queries,
    generate_gap_query,
)
from gate import is_sufficient
from web_fallback_retriever import WebFallbackRetriever

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
JSON_DIR = os.path.join(PROJECT_ROOT, "data")
MC_PATH = os.path.join(JSON_DIR, "ADRD_Caregiving_Multiple_Choice.json")
TF_PATH = os.path.join(JSON_DIR, "ADRD_Caregiving_True_or_False.json")


def load_adrd_bench_questions(subset="all"):
    records = []
    if subset in ("mc", "all") and os.path.exists(MC_PATH):
        with open(MC_PATH, encoding="utf-8") as f:
            for item in json.load(f)["data"]:
                options = item.get("Options", {})
                ans_letter = item["Answer"]
                query = f"{item['Question']}\nOptions:\n" + "\n".join([f"  {k}. {v}" for k, v in options.items()])
                records.append({
                    "Question_ID": f"ADRD_MC_{item['ID']:03d}",
                    "Type": "MC",
                    "Question": query,
                    "Stem": item["Question"],
                    "Options_Dict": options,
                    "Ground_Truth_Answer": options.get(ans_letter, ans_letter),
                    "Correct_Letter": ans_letter,
                })
    if subset in ("tf", "all") and os.path.exists(TF_PATH):
        with open(TF_PATH, encoding="utf-8") as f:
            for item in json.load(f)["data"]:
                records.append({
                    "Question_ID": f"ADRD_TF_{item['ID']:03d}",
                    "Type": "TF",
                    "Question": item["Question"],
                    "Ground_Truth_Answer": item["Answer"],
                    "Correct_Letter": item["Answer"],
                })
    return records


def clean_json_string(raw_str):
    s = (raw_str or "").strip()
    if s.startswith("```json"): 
        s = s[7:]
    elif s.startswith("```"): 
        s = s[3:]
    if s.endswith("```"): 
        s = s[:-3]
    return s.strip()


def main():
    parser = argparse.ArgumentParser(description="Batch retrieval for ADRD-Bench (Sentence-Level CRAG)")
    parser.add_argument("--top_k", type=int, default=getattr(config, "retrieval_top_k", 20), help="Final passages to return per question")
    parser.add_argument("--pre_k", type=int, default=getattr(config, "retrieval_pre_k", 100), help="Candidates before reranking")
    parser.add_argument("--window", type=int, default=getattr(config, "retrieval_window_size", 800), help="Context window expansion")
    parser.add_argument("--subset", choices=["mc", "tf", "all"], default=getattr(config, "default_subset", "all"), help="Subset to retrieve")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated Question_IDs to retrieve (subset for fast validation). Overrides --subset filtering to these IDs.")
    args = parser.parse_args()

    print("🔧 Initializing AdvancedRetriever (Local)...")
    retriever = AdvancedRetriever()

    web_retriever = WebFallbackRetriever(
        allow_domains=getattr(config, "web_allow_domains", []),
        cache_dir=getattr(config, "web_cache_dir", os.path.join(PROJECT_ROOT, "knowledge_base", "web_cache")),
        timeout_sec=getattr(config, "web_timeout_sec", 20),
        sleep_sec=getattr(config, "web_sleep_sec", 0.2),
        domain_mode=getattr(config, "web_domain_mode", "allowlist"),
        block_domains=getattr(config, "web_block_domains", []),
    )

    questions = load_adrd_bench_questions(subset=args.subset)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        questions = [q for q in questions if q["Question_ID"] in wanted]
        print(f"🎯 Subset filter: {len(questions)} of requested {len(wanted)} IDs matched.")
    os.makedirs(os.path.join(PROJECT_ROOT, "retrieval_results"), exist_ok=True)
    output_path = os.path.join(PROJECT_ROOT, "retrieval_results", f"retrieval_ADRD_{args.subset}_LOCAL_WEB_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    results = []
    checkpoint_every = getattr(config, "checkpoint_every", 5)

    def save_results():
        # Atomic-ish write: dump to a temp file then replace, so a crash mid-write
        # never corrupts the checkpoint.
        tmp_path = output_path + ".tmp"
        pd.DataFrame(results).to_csv(tmp_path, index=False, encoding="utf-8-sig")
        os.replace(tmp_path, output_path)

    for idx, item in enumerate(tqdm(questions, desc="Retrieving")):
        q_type = item["Type"]
        original_question = item["Question"]
        question_id = item["Question_ID"]

        queries = []
        retrieved_contexts, sources, scores = [], [], []
        retrieved_chunk_ids = set()

        # STEP 1: Query rewrite
        if q_type == "TF":
            try:
                rw = json.loads(clean_json_string(rewrite_tf_query(original_question)))
                queries = [q for q in rw.get("queries", []) if q and q != original_question] or [original_question]
            except Exception:
                queries = [original_question]
        elif q_type == "MC":
            try:
                decomp = json.loads(clean_json_string(decompose_mc_options(item.get("Stem", ""), item.get("Options_Dict", {}))))
                queries = [q for q in list(decomp.get("option_queries", {}).values()) if q] or [original_question]
            except Exception:
                queries = [original_question]

        # STEP 1.5: FLARE/HyDE draft-then-retrieve — add hypothetical source-style answer
        # sentences as extra queries to surface specific facts the question-query misses.
        if getattr(config, "draft_retrieve_enabled", False):
            try:
                stem = item.get("Stem", "") if q_type == "MC" else original_question
                draft_raw = generate_draft_query(stem, q_type, item.get("Options_Dict", {}))
                draft_qs = [q for q in json.loads(clean_json_string(draft_raw)).get("queries", []) if q and q not in queries]
                queries = queries + draft_qs
            except Exception:
                pass

        # STEP 2: Local retrieval pooling
        all_pool = []
        for q in queries:
            passages, chunk_scores, chunk_sources = retriever.get_retrieved_passages(
                q, top_k=args.top_k,
                bm25_weight=getattr(config, "bm25_weight", 0.30),
                vector_weight=getattr(config, "vector_weight", 0.70),
                pre_k=args.pre_k, window_size=args.window,
            )
            for p, s, src in zip(passages, chunk_scores, chunk_sources):
                all_pool.append((p, s, src))

        all_pool.sort(key=lambda x: x[1], reverse=True)

        # STEP 3: Local cutoff
        for p, s, src in all_pool:
            if len(retrieved_contexts) >= args.top_k: break
            pid = passage_id(src, p)
            if pid not in retrieved_chunk_ids:
                retrieved_contexts.append(p)
                sources.append(src)
                scores.append(s)
                retrieved_chunk_ids.add(pid)

        # STEP 4: sufficiency GATE — NLI (TF) / ItV (MC), replacing the GPT-4o evaluator
        tf_verdict, tf_confidence, eval_missing, raw_eval = None, None, "", None
        satisfied = is_sufficient(q_type, original_question, retrieved_contexts) if retrieved_contexts else False

        # STEP 4.5: Web Fallback (Sentence-Level CRAG Filter)
        web_used = False
        web_queries_used = []
        web_sources_used = []

        web_enabled = getattr(config, "web_enabled", False)
        web_rounds = getattr(config, "web_max_rounds", 2)
        verify_below = getattr(config, "tf_web_verify_below", "high")

        # Gate is now the calibrated NLI/ItV sufficiency decision, so a single bool suffices.
        def need_web():
            return not satisfied

        # STEP 4.25: GAP-GUIDED LOCAL RE-RETRIEVAL (self-contained; tried BEFORE web).
        # The answer is often already in the local KB but was buried by the generic query;
        # re-query the KB with the specific missing fact and re-evaluate.
        gap_local_used = False
        if need_web() and getattr(config, "gap_local_enabled", True):
            try:
                gq = generate_gap_query(original_question, eval_missing)
                if gq:
                    gp, _, gsrc = retriever.get_retrieved_passages(
                        gq, top_k=args.top_k,
                        bm25_weight=getattr(config, "bm25_weight", 0.30),
                        vector_weight=getattr(config, "vector_weight", 0.70),
                        pre_k=args.pre_k, window_size=args.window)
                    if gp and hasattr(retriever, "rerank_texts"):
                        rp, rs, rl = retriever.rerank_texts(original_question, retrieved_contexts + gp, sources + gsrc)
                        nc, ns, nsc, seen = [], [], [], set()
                        for p, s, lg in zip(rp, rs, rl):
                            if len(nc) >= args.top_k: break
                            pid = passage_id(s, p)
                            if pid in seen: continue
                            seen.add(pid); nc.append(p); ns.append(s); nsc.append(logit_to_prob(lg))
                        retrieved_contexts, sources, scores = nc, ns, nsc
                        gap_local_used = True
                        satisfied, tf_verdict, tf_confidence, eval_missing, _ = run_eval(
                            q_type, original_question, retrieved_contexts)
            except Exception:
                pass

        agentic = getattr(config, "agentic_web_enabled", False)
        if need_web() and web_enabled:
            for _round in range(web_rounds):
                if agentic:
                    # Agentic: reflect on what's already retrieved + what's missing, then
                    # issue refined/diverse queries for THIS round (different angle each round).
                    try:
                        wq_raw = agentic_search_queries(original_question, retrieved_contexts, eval_missing, q_type, _round)
                        wq_res = json.loads(clean_json_string(wq_raw))
                        generated_queries = [q for wq in wq_res.get("queries", []) if (q := wq.strip())]
                        # round 0 also seeds with the original sub-queries; later rounds rely on fresh angles
                        web_queries_used = generated_queries + ([q for q in queries if q not in generated_queries] if _round == 0 else [])
                        if not web_queries_used:
                            web_queries_used = list(queries)
                    except Exception:
                        web_queries_used = list(queries)
                elif eval_missing:
                    try:
                        wq_raw = generate_web_queries_from_missing_info(original_question, eval_missing, q_type)
                        wq_res = json.loads(clean_json_string(wq_raw))
                        generated_queries = [q for wq in wq_res.get("queries", []) if (q := wq.strip())]
                        web_queries_used = generated_queries + [q for q in queries if q not in generated_queries]
                    except Exception:
                        web_queries_used = list(queries)
                else:
                    web_queries_used = list(queries)

                # Fetch high-purity external "sentence clusters"
                wp, ws = web_retriever.retrieve(
                    queries=web_queries_used,
                    per_query_k=getattr(config, "web_per_query_k", 5),
                    max_page_chars=getattr(config, "web_max_page_chars", 25000),
                    max_sentences_per_source=getattr(config, "web_max_sentences_per_source", 12),
                )

                if not wp: break

                # Web relevance pre-filter: keep only the top-N web sentences most relevant
                # to the question before merging, dropping the marginally-relevant tail.
                prefilter_keep = getattr(config, "web_prefilter_keep", 0)
                if prefilter_keep and len(wp) > prefilter_keep and hasattr(retriever, "rerank_texts"):
                    wp_s, ws_s, _ = retriever.rerank_texts(original_question, wp, ws)
                    wp, ws = wp_s[:prefilter_keep], ws_s[:prefilter_keep]

                web_used = True
                web_sources_used.extend(ws)

                # Merge complete local chunks with pure external sentences
                merged_passages = retrieved_contexts + wp
                merged_sources = sources + ws

                # [Applied Inspiration]: Use Cross-Encoder for global, sentence-level penetrating reranking
                if hasattr(retriever, "rerank_texts"):
                    reranked_passages, reranked_sources, reranked_logits = retriever.rerank_texts(
                        original_question, merged_passages, merged_sources
                    )

                    # Reserve at least `local_floor` of the top_k slots for local passages
                    # so high-scoring web sentences can't completely flood out the local
                    # answer context. Web above the cap is deferred and only used to fill
                    # leftover slots if local runs out.
                    local_floor = getattr(config, "web_final_local_floor", 0)
                    web_cap = max(0, args.top_k - local_floor)
                    retrieved_contexts, sources, scores, retrieved_chunk_ids = [], [], [], set()
                    web_count, deferred_web = 0, []
                    for p, src, lg in zip(reranked_passages, reranked_sources, reranked_logits):
                        if len(retrieved_contexts) >= args.top_k: break
                        pid = passage_id(src, p)
                        if pid in retrieved_chunk_ids: continue
                        retrieved_chunk_ids.add(pid)
                        is_web = str(src).startswith("http")
                        if is_web and web_count >= web_cap:
                            deferred_web.append((p, src, lg))   # over cap: defer
                            continue
                        retrieved_contexts.append(p)
                        sources.append(src)
                        scores.append(logit_to_prob(lg))
                        if is_web:
                            web_count += 1
                    # Fill any remaining slots with the deferred (over-cap) web passages.
                    for p, src, lg in deferred_web:
                        if len(retrieved_contexts) >= args.top_k: break
                        retrieved_contexts.append(p)
                        sources.append(src)
                        scores.append(logit_to_prob(lg))

                # Re-evaluate after Web fallback
                raw_eval = None
                eval_missing = ""

                satisfied, tf_verdict, tf_confidence, eval_missing, _ = run_eval(q_type, original_question, retrieved_contexts)

                # Stop early once the gate says the context is sufficient.
                if not need_web():
                    break

        print(f"📄 [{question_id}] passages={len(retrieved_contexts)} satisfied={satisfied}" + (f" tf_verdict={tf_verdict}({tf_confidence})" if q_type == "TF" else "") + f" gap_local={gap_local_used} web_used={web_used}")

        results.append({
            "Question_ID": question_id, "Type": q_type,
            "Question": original_question, "Ground_Truth_Answer": item["Ground_Truth_Answer"],
            "Correct_Letter": item["Correct_Letter"],
            "Retrieved_Passages": json.dumps(retrieved_contexts, ensure_ascii=False),
            "Retrieved_Sources": json.dumps(sources, ensure_ascii=False),
            "Satisfied": satisfied, "TF_Verdict": tf_verdict, "TF_Confidence": tf_confidence,
            "Gap_Local_Used": gap_local_used,
            "Eval_Missing_Information": eval_missing, "Web_Used": web_used,
            "Web_Queries": json.dumps(web_queries_used, ensure_ascii=False),
            "Web_Sources": json.dumps(list(dict.fromkeys(web_sources_used))[:30], ensure_ascii=False),
        })

        # Incremental checkpoint so a mid-run crash (esp. with web fallback enabled)
        # does not lose all completed questions.
        if checkpoint_every and (idx + 1) % checkpoint_every == 0:
            save_results()

    save_results()
    print(f"\n✅ Sentence-Level CRAG Retrieval Complete! Saved to {output_path}")


if __name__ == "__main__":
    main()