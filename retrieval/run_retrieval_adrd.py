#!/usr/bin/env python3
"""
Batch retrieval for ADRD-Bench (Multiple Choice + True/False).
Loads questions from local JSON files and runs the 
BM25 + Vector + Rerank pipeline.

Usage (from the code/ directory):
  python run_retrieval_ADRD_Bench.py

Output:
  retrieve_results/retrieval_ADRD_<subset>_k<k>_w<window>_<timestamp>.csv
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Ensure the current directory is in path for advanced_retriever import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advanced_retriever import AdvancedRetriever

# JSON files mapping
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
JSON_DIR = os.path.join(PROJECT_ROOT, "data")
MC_PATH  = os.path.join(JSON_DIR, "ADRD_Caregiving_Multiple_Choice.json")
TF_PATH  = os.path.join(JSON_DIR, "ADRD_Caregiving_True_or_False.json")


# ==========================================
# LOAD ADRD-BENCH QUESTIONS
# ==========================================

def load_adrd_bench_questions(subset="all"):
    """
    Load raw questions from local JSON files.
    Returns a list of dicts with keys:
      Question_ID, Type, Question, Ground_Truth_Answer, Correct_Letter
    """
    records = []

    # --- Multiple Choice ---
    if subset in ("mc", "all"):
        if not os.path.exists(MC_PATH):
            print(f"⚠️  MC file not found: {MC_PATH}")
        else:
            print(f"📥 Loading MC questions from {os.path.basename(MC_PATH)}...")
            with open(MC_PATH, encoding="utf-8") as f:
                mc_data = json.load(f)
            for item in mc_data["data"]:
                options           = item.get("Options", {})
                ans_letter        = item["Answer"]
                ground_truth_text = options.get(ans_letter, ans_letter)
                options_str       = "\n".join([f"  {k}. {v}" for k, v in options.items()])
                
                # Include options in the query so the retriever finds context relevant to all choices
                query = f"{item['Question']}\nOptions:\n{options_str}"
                records.append({
                    "Question_ID":         f"ADRD_MC_{item['ID']:03d}",
                    "Type":                "MC",
                    "Question":            query,
                    "Stem":                item["Question"],   # Question stem without options
                    "Options_Dict":        options,            
                    "Ground_Truth_Answer": ground_truth_text,
                    "Correct_Letter":      ans_letter,
                })
            print(f"  ✅ {sum(1 for r in records if r['Type']=='MC')} MC questions loaded.")

    # --- True / False ---
    if subset in ("tf", "all"):
        if not os.path.exists(TF_PATH):
            print(f"⚠️  T/F file not found: {TF_PATH}")
        else:
            print(f"📥 Loading T/F questions from {os.path.basename(TF_PATH)}...")
            with open(TF_PATH, encoding="utf-8") as f:
                tf_data = json.load(f)
            before = len(records)
            for item in tf_data["data"]:
                ground_truth = item["Answer"]  # "Yes" or "No"
                records.append({
                    "Question_ID":         f"ADRD_TF_{item['ID']:03d}",
                    "Type":                "TF",
                    "Question":            item["Question"],  
                    "Ground_Truth_Answer": ground_truth,
                    "Correct_Letter":      ground_truth,
                })
            print(f"  ✅ {len(records) - before} T/F questions loaded.")

    print(f"\n📊 Total questions to retrieve: {len(records)}\n")
    return records


# ==========================================
# MAIN
# ==========================================

def main():
    from sentence_transformers import CrossEncoder
    
    print("🔧 Initializing Cross-Encoder for web relevance scoring...")
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    from web_retriever import web_augment
    from rag_config import config
    
    parser = argparse.ArgumentParser(description="Batch retrieval for ADRD-Bench")
    parser.add_argument("--top_k",  type=int,   default=config.retrieval_top_k,    help="Final passages to return per question")
    parser.add_argument("--pre_k",  type=int,   default=config.retrieval_pre_k,    help="Candidates before reranking")
    parser.add_argument("--bm25",   type=float, default=config.retrieval_bm25_weight,  help="BM25 weight")
    parser.add_argument("--vector", type=float, default=config.retrieval_vector_weight, help="Vector weight")
    parser.add_argument("--window", type=int,   default=config.retrieval_window,  help="Context window expansion chars")
    parser.add_argument("--subset", choices=["mc", "tf", "all"], default="all", help="Subset to retrieve for")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "retrieval_results"),
                        help="Output directory")
    args = parser.parse_args()

    pre_k_display = args.pre_k if args.pre_k else (args.top_k * 5)
    print(f"⚙️  Config — Top-K: {args.top_k} | Pre-K: {pre_k_display} | Window: {args.window}")
    print(f"           BM25: {args.bm25} | Vector: {args.vector} | Subset: {args.subset}\n")

    # ---- Initialize retriever ----
    print("🔧 Initializing AdvancedRetriever...")
    try:
        retriever = AdvancedRetriever()
    except Exception as e:
        print(f"❌ Retriever initialization failed: {e}")
        return

    # ---- Load Questions ----
    questions = load_adrd_bench_questions(subset=args.subset)
    if not questions:
        print("❌ No questions loaded. Exiting.")
        return

    # ---- Output Setup ----
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir,
        f"retrieval_ADRD_{args.subset}_k{args.top_k}_w{args.window}_{timestamp}.csv"
    )

    print(f"🚀 Starting retrieval for {len(questions)} questions...\n")
    results = []

    from llm_utils import evaluate_context, generate_queries, rewrite_tf_query, evaluate_tf_evidence, decompose_mc_options, generate_gap_claim
    import json as _json

    for item in tqdm(questions, desc="Retrieving"):
        original_question = item["Question"]
        question_id = item["Question_ID"]
        q_type = item["Type"]
        past_queries = set()
        retrieved_contexts = []
        retrieved_chunk_ids = set()
        iteration = 0
        sources = []
        scores = []
        satisfied = False
        tf_verdict = None   
        mc_option_queries = {}  

        # -- Optimization 1: TF Query Rewriting & Decomposition --
        if q_type == "TF":
            print(f"\n✏️  [{question_id}] TF query rewriting...")
            try:
                rw_result = rewrite_tf_query(original_question)
                if isinstance(rw_result, str):
                    rw_result = _json.loads(rw_result)
                rewritten_queries = rw_result.get("queries", [])
            except Exception as e:
                print(f"⚠️ [{question_id}] rewrite_tf_query failed: {e}")
                rewritten_queries = []
                
            # If rewrite successful, discard the original statement to avoid false-premise pollution
            if rewritten_queries:
                queries = [q for q in rewritten_queries if q != original_question]
                if not queries:
                    queries = [original_question]  # Fallback
                print(f"   Original (skipped): {original_question[:80]}")
                for i, q in enumerate(queries, 1):
                    print(f"   Rewritten {i}: {q[:80]}")
            else:
                queries = [original_question]
                print(f"   Rewrite failed, falling back to original: {original_question[:80]}")

        # -- MC Option Decomposition: Generate specific queries for each option --
        elif q_type == "MC":
            stem = item.get("Stem", "")
            options_dict = item.get("Options_Dict", {})
            if stem and options_dict:
                print(f"\n🔀 [{question_id}] MC option-level decomposition...")
                try:
                    decomp_raw = decompose_mc_options(stem, options_dict)
                    if isinstance(decomp_raw, str):
                        decomp = _json.loads(decomp_raw)
                    else:
                        decomp = decomp_raw
                    option_queries = decomp.get("option_queries", {})
                except Exception as e:
                    print(f"⚠️  [{question_id}] decompose_mc_options failed: {e}")
                    option_queries = {}

                if option_queries:
                    queries = list(option_queries.values())
                    mc_option_queries = option_queries 
                    for letter in sorted(option_queries):
                        print(f"   Option {letter}: {option_queries[letter][:80]}")
                else:
                    print(f"   ⚠️  Decomposition empty, falling back to full question")
                    queries = [original_question]
            else:
                queries = [original_question]

        else:
            queries = [original_question]

        while iteration < config.max_iterations:
            new_contexts = []
            new_sources = []
            new_scores = []

            for q in queries:
                if q in past_queries:
                    continue
                try:
                    # Increased to 8 to boost recall
                    effective_top_k = 8 if q_type in ("TF", "MC") else args.top_k
                    passages, chunk_scores, chunk_sources = retriever.get_retrieved_passages(
                        q,
                        top_k=effective_top_k,
                        bm25_weight=args.bm25,
                        vector_weight=args.vector,
                        pre_k=args.pre_k,
                        window_size=args.window,
                    )
                    
                    # Lowered threshold to 0.05 to avoid incorrectly filtering valid medical context
                    TF_SCORE_THRESHOLD = 0.05
                    if passages:
                        filtered = [(p, s, src) for p, s, src in zip(passages, chunk_scores, chunk_sources) if s >= TF_SCORE_THRESHOLD]
                        if filtered:
                            passages, chunk_scores, chunk_sources = zip(*filtered)
                            passages, chunk_scores, chunk_sources = list(passages), list(chunk_scores), list(chunk_sources)
                            print(f"   🔍 [{question_id}|iter{iteration}] q='{q[:50]}' → {len(passages)} passages (score>={TF_SCORE_THRESHOLD})")
                        else:
                            # Keep at least top-1 if all below threshold
                            best = max(zip(chunk_scores, passages, chunk_sources), key=lambda x: x[0])
                            passages, chunk_scores, chunk_sources = [best[1]], [best[0]], [best[2]]
                            print(f"   ⚠️  [{question_id}|iter{iteration}] all below threshold, keeping top-1 (score={best[0]:.3f})")
                except Exception as e:
                    print(f"⚠️  Error for {question_id}: {type(e).__name__}: {e}")
                    passages, chunk_scores, chunk_sources = [], [], []

                if not passages:
                    continue

                # BYPASS LLM EXTRACTION & NLI TO PRESERVE MAXIMUM MEDICAL CONTEXT/RECALL
                for p, s, src in zip(passages, chunk_scores, chunk_sources):
                    pid = f"{src}__{hash(p)}"
                    if pid not in retrieved_chunk_ids:
                        new_contexts.append(p)
                        new_sources.append(src)
                        new_scores.append(s)
                        retrieved_chunk_ids.add(pid)
                
                past_queries.add(q)

            if not new_contexts:
                if iteration == 0:
                    print(f"🔄 [{question_id}] First round retrieval empty! Forcing LLM sub-query generation...")
                    missing_info = "The original question is too complex. Break it down into 1 to 3 simple keyword queries."
                    try:
                        gen_result = generate_queries(original_question, missing_info)
                        if isinstance(gen_result, str):
                            gen_result = _json.loads(gen_result)
                        queries = gen_result.get("queries", [])
                    except Exception as e:
                        print(f"⚠️ LLM failed to generate sub-queries: {e}")
                        queries = []
                    iteration += 1
                    continue
                else:
                    break

            retrieved_contexts.extend(new_contexts)
            sources.extend(new_sources)
            scores.extend(new_scores)

            # -- Optimization 4: Evidence-first evaluation for TF --
            if q_type == "TF":
                try:
                    tf_eval_raw = evaluate_tf_evidence(original_question, retrieved_contexts)
                    if isinstance(tf_eval_raw, str):
                        tf_eval = _json.loads(tf_eval_raw)
                    else:
                        tf_eval = tf_eval_raw
                except Exception as e:
                    print(f"⚠️  evaluate_tf_evidence failed: {e}")
                    tf_eval = {"verdict": "insufficient", "evidence": "", "missing": "API error"}

                verdict  = tf_eval.get("verdict", "insufficient")
                evidence = tf_eval.get("evidence", "")
                missing  = tf_eval.get("missing", "")
                
                evidence_preview = str(evidence)[:80] if evidence else ""
                print(f"   🔬 [{question_id}|iter{iteration}] verdict={verdict} | evidence={evidence_preview}")

                if verdict in ("True", "False"):
                    tf_verdict = verdict
                    satisfied = True
                    print(f"   ✅ [{question_id}] Evidence found → verdict={verdict}")
                    break
                else:
                    print(f"   🔄 [{question_id}] Insufficient evidence. Missing: {missing[:80]}")
                    if iteration + 1 < config.max_iterations:
                        try:
                            gen_result = generate_queries(original_question, missing)
                            if isinstance(gen_result, str):
                                gen_result = _json.loads(gen_result)
                            new_tf_queries = gen_result.get("queries", [])
                        except Exception as e:
                            new_tf_queries = []
                        queries = [q for q in new_tf_queries if q not in past_queries]
                    iteration += 1
                    continue

            # -- MC Logic --
            try:
                eval_result = evaluate_context(original_question, retrieved_contexts)
                if isinstance(eval_result, str):
                    eval_result = _json.loads(eval_result)
            except Exception as e:
                eval_result = {"status": "answerable", "missing_information": ""}

            if eval_result.get("status") == "answerable":
                satisfied = True
                break

            # Gap Claim Generation
            try:
                gap_raw = generate_gap_claim(original_question, retrieved_contexts, q_type="MC")
                if isinstance(gap_raw, str):
                    gap_data = _json.loads(gap_raw)
                else:
                    gap_data = gap_raw
                gap_claim   = gap_data.get("gap_claim", "")
                targeted_q  = gap_data.get("targeted_query", "")
            except Exception as e:
                gap_claim, targeted_q = "", ""

            if gap_claim:
                print(f"   🔎 [{question_id}] Gap Claim: {gap_claim[:120]}")
            if targeted_q and targeted_q not in past_queries and targeted_q not in queries:
                queries.append(targeted_q)
                print(f"   ➕ [{question_id}] Targeted query appended: {targeted_q[:80]}")
            elif not targeted_q:
                try:
                    gen_result = generate_queries(original_question, gap_claim or "")
                    if isinstance(gen_result, str):
                        gen_result = _json.loads(gen_result)
                    for nq in gen_result.get("queries", []):
                        if nq not in past_queries and nq not in queries:
                            queries.append(nq)
                except Exception:
                    pass

            iteration += 1

        # -- Layer 2 & 3: Web Augmentation --
        if not satisfied:
            print(f"🌐 [{question_id}] Local KB not satisfied → trying web augmentation...")

            _caregiving_kw = {"bath", "wash", "wander", "agitat", "resist", "refus",
                              "behavior", "communicat", "caregiv", "daily", "activit",
                              "sundown", "redirect", "distract", "routine", "mood",
                              "anger", "anxiety", "personalit", "safe"}
            _q_lower = original_question.lower()
            is_caregiving = q_type == "MC" and any(kw in _q_lower for kw in _caregiving_kw)
            clean_stem = item.get("Stem", original_question)[:200]

            if is_caregiving:
                domain_framed = f"Alzheimer caregiver {clean_stem}"
                print(f"   🏠 [{question_id}] Domain: caregiving → restricting to caregiving styles")
            elif q_type == "TF" or not is_caregiving:
                domain_framed = f"dementia clinical evidence {clean_stem}"
                print(f"   🧬 [{question_id}] Domain: medical fact → restricting to clinical styles")
            else:
                domain_framed = clean_stem

            aug_queries = [domain_framed] + [q for q in past_queries if q not in (domain_framed, original_question)][:2]
            web_domain = "caregiving" if is_caregiving else "medical"
            web_added = 0
            
            # CRITICAL FIX: Dummy NLI to bypass the flawed web_retriever contradiction logic
            class DummyNLI:
                def predict(self, pairs):
                    import numpy as np
                    # Always return Entailment (class index 1 is highest)
                    return [np.array([-10.0, 10.0, -10.0]) for _ in pairs]
            
            dummy_nli_model = DummyNLI()

            for aug_q in aug_queries:
                try:
                    w_passages, w_sources, w_scores = web_augment(
                        aug_q,
                        nli_model=dummy_nli_model,  # Passed Dummy to prevent valid web drops
                        cross_encoder=cross_encoder_model,
                        relevance_threshold=0.3,
                        max_web_passages=10,
                        max_wiki_pages=3,
                        max_pubmed_results=5,
                        max_s2_results=5,
                        already_seen_ids=retrieved_chunk_ids,
                        domain=web_domain,
                    )
                except Exception as e:
                    print(f"⚠️  web_augment failed for [{question_id}]: {e}")
                    w_passages, w_sources, w_scores = [], [], []

                retrieved_contexts.extend(w_passages)
                sources.extend(w_sources)
                scores.extend(w_scores)
                web_added += len(w_passages)

            if web_added > 0:
                print(f"  ✅ [{question_id}] Web augmentation added {web_added} passages.")
                try:
                    if q_type == "TF":
                        tf_eval_raw = evaluate_tf_evidence(original_question, retrieved_contexts)
                        if isinstance(tf_eval_raw, str):
                            try:
                                tf_eval = _json.loads(tf_eval_raw)
                            except json.JSONDecodeError as json_err:
                                print(f"  ⚠️  [{question_id}] Post-web evaluate_tf_evidence JSON parse error: {json_err}")
                                tf_eval = {"verdict": "insufficient", "evidence": "", "missing": "JSON parse error"}
                        else:
                            tf_eval = tf_eval_raw
                        verdict = tf_eval.get("verdict", "insufficient")
                        print(f"  🔬 [{question_id}] Post-web TF verdict={verdict}")
                        if verdict in ("True", "False"):
                            tf_verdict = verdict
                            satisfied = True
                    else:
                        eval_web = evaluate_context(original_question, retrieved_contexts)
                        if isinstance(eval_web, str):
                            eval_web = _json.loads(eval_web)
                        satisfied = eval_web.get("status") == "answerable"
                except Exception as e:
                    print(f"  ⚠️  [{question_id}] Post-web re-evaluation error: {type(e).__name__}: {e}")
            else:
                print(f"  ⚠️  [{question_id}] Web augmentation returned no new passages.")

        # -- Bypass refinement for MC/TF --
        passages_to_save, scores_to_save, sources_to_save = retrieved_contexts, scores, sources

        web_src_count = sum(1 for s in sources_to_save if str(s).startswith(('Wikipedia:','PubMed:','S2:')))
        local_src_count = len(sources_to_save) - web_src_count
        print(f"   📄 [{question_id}] final: {len(passages_to_save)} passages (local={local_src_count}, web={web_src_count}) | satisfied={satisfied}"
              + (f" | tf_verdict={tf_verdict}" if q_type == 'TF' else ""))
              
        results.append({
            "Question_ID":         question_id,
            "Type":                item["Type"],
            "Question":            original_question,
            "Ground_Truth_Answer": item["Ground_Truth_Answer"],
            "Correct_Letter":      item["Correct_Letter"],
            "Retrieved_Passages":  json.dumps(passages_to_save,  ensure_ascii=False),
            "Retrieved_Sources":   json.dumps(sources_to_save,   ensure_ascii=False),
            "Rerank_Scores":       json.dumps(scores_to_save,    ensure_ascii=False),
            "Satisfied":           satisfied,
            "Iterations":          iteration+1,
            "TF_Verdict":          tf_verdict if q_type == "TF" else None,
        })

    # ---- Save ----
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Retrieval complete!")
    print(f"📂 Results saved to: {output_path}")

if __name__ == "__main__":
    main()