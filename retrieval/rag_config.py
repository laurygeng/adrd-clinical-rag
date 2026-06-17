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

    # ==========================================
    # 2. Retrieval Parameters (run_retrieval_adrd.py)
    # ==========================================
    retrieval_top_k = 20
    retrieval_pre_k = 100
    retrieval_window_size = 800
    bm25_weight = 0.3
    vector_weight = 0.7
    default_subset = "all"  # choices: "mc", "tf", "all"

    # ==========================================
    # 3. Reranking Parameters (advanced_retriever.py)
    # ==========================================
    rerank_enabled = True
    # rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    # rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2' # 维持较快速度的重排模型
    # rag_config.py
    
    rerank_model_name = 'BAAI/bge-reranker-v2-m3'
    
    @property
    def rerank_device(self):
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'

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

    web_trigger_min_local_passages = 3

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
    ]

config = RAGConfig()