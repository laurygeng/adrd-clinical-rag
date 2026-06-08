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
    retrieval_pre_k = 25
    retrieval_window_size = 800
    bm25_weight = 0.65
    vector_weight = 0.35
    default_subset = "all"  # choices: "mc", "tf", "all"

    # ==========================================
    # 3. Reranking Parameters (advanced_retriever.py)
    # ==========================================
    rerank_enabled = True
    # rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2' # 维持较快速度的重排模型
    # rag_config.py
    
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

config = RAGConfig()