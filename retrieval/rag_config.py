# rag_config.py
import os
import warnings

# Disable Hugging Face tokenizers parallelism
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

class RAGConfig:
    # ==========================================
    # 1. Ingestion & Chunking Parameters (ingest_documents.py)
    # ==========================================
    parent_chunk_size = 800
    parent_chunk_overlap = 100
    child_chunk_size = 250
    child_chunk_overlap = 40
    # embed_model_name = 'all-MiniLM-L6-v2'
    embed_model_name = 'BAAI/bge-large-en-v1.5'  # 换成高维度的、更强大的向量模型
    chroma_batch_size = 500

    # BGE embeddings work best with L2-normalized vectors + cosine space, and with
    # a query-side instruction prefix (passages are embedded WITHOUT any instruction).
    # NOTE: changing embed_normalize / embed_space only takes effect after re-ingesting
    # (the vector DB must be rebuilt so stored vectors match the query-side settings).
    embed_normalize = True
    embed_space = "cosine"
    bge_query_instruction = "Represent this sentence for searching relevant passages: "

    # ==========================================
    # 2. Retrieval Parameters (run_retrieval_adrd.py)
    # ==========================================
    retrieval_top_k = 20
    retrieval_pre_k = 100
    retrieval_window_size = 800
    bm25_weight = 0.3
    vector_weight = 0.7
    default_subset = "all"  # choices: "mc", "tf", "all"
    checkpoint_every = 20    # write a partial CSV every N questions (crash safety / progress review)

    # ==========================================
    # 3. Reranking Parameters (advanced_retriever.py)
    # ==========================================
    rerank_enabled = True
    # rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    # rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2' # 维持较快速度的重排模型
    # rag_config.py
    
    rerank_model_name = 'BAAI/bge-reranker-v2-m3'
    rerank_max_chars = 1000   # max chars of each passage fed to the cross-encoder
    rerank_min_prob = 0.05    # drop reranked passages below this probability

    _rerank_device_cache = None

    @property
    def rerank_device(self):
        # Cache the (expensive) torch import + cuda probe; this property is read
        # many times per run.
        if RAGConfig._rerank_device_cache is None:
            try:
                import torch
                RAGConfig._rerank_device_cache = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                RAGConfig._rerank_device_cache = 'cpu'
        return RAGConfig._rerank_device_cache

    # ==========================================
    # 4. LLM & Evaluation Parameters (llm_utils.py)
    # ==========================================
    llm_eval_model = "gpt-4o"
    llm_rewrite_model = "gpt-4o"
    
    
    # ==========================================
    # 5. Web fallback retrieval (FREE, research-only)
    # ==========================================
    web_enabled = True
    web_max_rounds = 2
    web_per_query_k = 5
    web_timeout_sec = 20
    web_sleep_sec = 0.2
    web_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../knowledge_base/web_cache"))

    web_max_page_chars = 25000
    web_chunk_chars = 2000
    web_chunk_overlap = 200

    # Cap sentence-level passages kept per web/PubMed source so the cross-encoder
    # rerank pool (local + web) does not explode on CPU.
    web_max_sentences_per_source = 12

    # Guard: after merging local+web and reranking, reserve at least this many of the
    # final top_k slots for LOCAL passages (when enough exist), so high-scoring web
    # sentences cannot completely flood out the answer-bearing local context.
    web_final_local_floor = 8

    web_trigger_min_local_passages = 3

    # Domain policy for web fallback:
    #   "allowlist" = only fetch from web_allow_domains (strict, original behavior)
    #   "blocklist" = fetch from ANY domain except web_block_domains (open; rely on the
    #                 cross-encoder rerank to reject topically-irrelevant noise)
    # NOTE: full "blocklist" (open) mode was tested and REGRESSED MC by -5 — open domains
    # flooded the rerank with marketing/SEO blogs that displaced good local context
    # (e.g. MC_026 ended up 0 local / 20 web). Reverted to a curated allowlist.
    web_domain_mode = "allowlist"
    web_block_domains = [
        # social / UGC
        "facebook.com", "m.facebook.com", "twitter.com", "x.com", "instagram.com",
        "tiktok.com", "youtube.com", "youtu.be", "reddit.com", "pinterest.com",
        "linkedin.com", "quora.com", "tumblr.com", "threads.net", "medium.com",
        # commerce / ads / SEO farms / doc dumps
        "amazon.com", "ebay.com", "walmart.com", "etsy.com", "yelp.com",
        "answers.com", "ehow.com", "slideshare.net", "scribd.com", "coursehero.com",
        "pinterest.co.uk",
    ]

    # TF web-verification gate: when a TF verdict IS reached but its confidence is below
    # this level, still trigger web verification + re-judge (instead of trusting it blindly).
    # One of: "high" | "medium" | "low". "high" => verify everything not high-confidence.
    tf_web_verify_below = "high"

    web_allow_domains = [
        "nia.nih.gov",
        "nih.gov",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "cdc.gov",
        "who.int",
        "alz.org",
        "alzheimers.gov",
        "en.wikipedia.org",
        "www.ncbi.nlm.nih.gov",
        "mayoclinic.org",       # 梅奥诊所（照护指南极佳）
        "clevelandclinic.org",  # 克利夫兰医学中心
        "hopkinsmedicine.org",  # 约翰霍普金斯
        "alzdiscovery.org",     # 阿尔茨海默症药物发现基金会
        "dementia.org",         # 痴呆症专题宣教网站
        # Curated reputable caregiving/health additions (vetted from the open-domain run):
        "alzheimers.org.uk",    # Alzheimer's Society UK
        "dementiauk.org",       # Dementia UK
        "nccdp.org",            # National Council of Certified Dementia Practitioners
        "nia.nih.gov",          # National Institute on Aging
        "agingcare.com",        # 照护问答（编辑审核）
        "dailycaring.com",      # 照护实操（编辑审核）
        "verywellhealth.com",   # 医学审核科普
        "healthline.com",       # 医学审核科普
        "nhs.uk",               # UK National Health Service
        "caregiver.org",        # Family Caregiver Alliance
    ]

config = RAGConfig()