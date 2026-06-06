# advanced_retriever.py
import os
import sys
import logging
import json
import pickle
import warnings
import re
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
from typing import List, Tuple, Optional

# Ignore LangChain warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

# Core dependency check
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from langchain_community.vectorstores import Chroma as LCChroma
    from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# BM25 & Ensemble compatibility
try:
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    BM25Retriever = None
    EnsembleRetriever = None

class Config:
    """Internal Configuration: Reranking and Hardware Acceleration"""
    rerank_enabled = True
    rerank_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    default_window_size = 500  # Default dynamic window expansion characters
    try:
        import torch
        rerank_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        rerank_device = 'cpu'

config_internal = Config()

class AdvancedRetriever:
    def score_text(self, query: str, text: str) -> float:
        """
        Score a single (query, text) pair using the cross-encoder. Used for sentence-level filtering.
        """
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(config_internal.rerank_model_name, device=config_internal.rerank_device)
        return float(self._cross_encoder.predict([[query, text[:1000]]])[0])
    def __init__(self):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path settings
        self.data_dir = os.path.join(os.path.abspath(os.path.join(current_script_dir, '../knowledge_base/advanced_rag')))
        self.chroma_persist_dir = os.path.join(os.path.abspath(os.path.join(current_script_dir, '../knowledge_base/braincheck_vectordb')))
        self.collection_name = "braincheck_advanced"

        self.children_pkl_path = os.path.join(self.data_dir, "vector_ready_children.pkl")
        self.parent_map_path = os.path.join(self.data_dir, "parent_map.json")

        # Component initialization
        self.chroma_collection = None
        self._embed_model = None
        self._cross_encoder = None
        
        # Optimization: Preload retriever components
        self.bm25_retriever = None
        self.vector_retriever_base = None
        
        self.child_chunks = []
        self.parent_map = {}
        
        self._load_data()
        self._init_chroma()
        self._init_retrievers() # Perform one-time initialization
        
    def _load_data(self):
        """Load data and force Key type alignment"""
        if os.path.exists(self.children_pkl_path):
            with open(self.children_pkl_path, 'rb') as f:
                self.child_chunks = pickle.load(f)
        else:
            raise FileNotFoundError(f"Missing {self.children_pkl_path}")
            
        if os.path.exists(self.parent_map_path):
            with open(self.parent_map_path, 'r', encoding='utf-8') as f:
                raw_map = json.load(f)
                self.parent_map = {str(k).strip(): v for k, v in raw_map.items()}
        else:
            logging.warning("⚠️ Parent map not found!")

    def _init_chroma(self):
        """Initialize ChromaDB (Chroma Native Client)"""
        client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        try:
            self.chroma_collection = client.get_collection(self.collection_name)
        except Exception:
            self.chroma_collection = client.create_collection(self.collection_name)
        
        # SentenceTransformer for generating Embeddings (Native)
        self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        if self.chroma_collection.count() == 0 and self.child_chunks:
            self._populate_chroma()

    def _init_retrievers(self):
        """
        [Performance Optimization]: Build retriever objects once during the initialization phase
        """
        print("🛠️ Preloading retrieval components...")
        
        # 1. Preload BM25
        if BM25Retriever and self.child_chunks:
            self.bm25_retriever = BM25Retriever.from_documents(self.child_chunks)
        
        # 2. Preload Vector Store LangChain Wrapper
        # Preload Embedding Model
        hf_embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        # Preload Chroma Object
        self.vector_store = LCChroma(
            persist_directory=self.chroma_persist_dir, 
            collection_name=self.collection_name, 
            embedding_function=hf_embeddings
        )
        print("✅ Retrieval components preloaded successfully.")

    def _populate_chroma(self):
        """Populate the vector database"""
        batch_size = 500
        texts = [d.page_content for d in self.child_chunks]
        embeddings = self._embed_model.encode(texts, show_progress_bar=True).tolist()
        ids = [str(d.metadata.get('child_id')) for d in self.child_chunks]
        metadatas = [d.metadata for d in self.child_chunks]
        
        for i in range(0, len(self.child_chunks), batch_size):
            end = min(i + batch_size, len(self.child_chunks))
            self.chroma_collection.add(
                ids=ids[i:end], documents=texts[i:end], embeddings=embeddings[i:end], metadatas=metadatas[i:end]
            )

    def _get_smart_window(self, parent_text: str, child_text: str, window_size: int) -> str:
        """Core Fix: Identify and extract content after 'Content:' tag for matching"""
        match_target = child_text
        if "Content:" in child_text:
            match_target = child_text.split("Content:")[-1].strip()
        
        query_pattern = re.escape(match_target).replace(r'\ ', r'\s+').replace(r'\n', r'\s+')
        match = re.search(query_pattern, parent_text)
        
        if not match:
            return child_text
            
        start_idx, end_idx = match.start(), match.end()
        ideal_start = max(0, start_idx - window_size)
        ideal_end = min(len(parent_text), end_idx + window_size)
        
        prefix = parent_text[ideal_start:start_idx]
        sentence_starts = list(re.finditer(r'[。！？\.!\?\n]\s*', prefix))
        actual_start = ideal_start + sentence_starts[-1].end() if sentence_starts else ideal_start
        
        suffix = parent_text[end_idx:ideal_end]
        sentence_end = re.search(r'[。！？\.!\?\n]', suffix)
        actual_end = end_idx + sentence_end.end() if sentence_end else ideal_end
        
        final_text = parent_text[actual_start:actual_end].strip()
        if actual_start > 0: final_text = "..." + final_text
        if actual_end < len(parent_text): final_text = final_text + "..."
        
        return final_text

    def _build_ensemble_retriever(self, search_k, bm25_weight=0.5, vector_weight=0.5):
        """
        [Optimized Version]: Directly use preloaded retrievers, only updating k value
        """
        retrievers = []
        
        # 1. Use preloaded BM25
        if self.bm25_retriever:
            self.bm25_retriever.k = search_k
            retrievers.append(self.bm25_retriever)
        
        # 2. Use preloaded Vector Store
        retrievers.append(self.vector_store.as_retriever(search_kwargs={'k': search_k}))
        
        if len(retrievers) > 1:
            return EnsembleRetriever(retrievers=retrievers, weights=[bm25_weight, vector_weight])
        return retrievers[0]

    def _rerank_passages(self, query, passages, sources, ids):
        if not config_internal.rerank_enabled: return passages, sources, ids, [1.0]*len(passages)
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(config_internal.rerank_model_name, device=config_internal.rerank_device)
        pairs = [[query, p[:1000]] for p in passages]
        scores = self._cross_encoder.predict(pairs).tolist()
        combined = sorted(zip(passages, sources, ids, scores), key=lambda x: x[3], reverse=True)
        return [x[0] for x in combined], [x[1] for x in combined], [x[2] for x in combined], [x[3] for x in combined]

    def get_retrieved_passages(self, question: str, top_k: int = 3, bm25_weight: float = 0.5, 
                               vector_weight: float = 0.5, pre_k: Optional[int] = None,
                               window_size: Optional[int] = None) -> Tuple[List[str], List[float], List[str]]:
        from rag_config import config
        w_size = window_size if window_size is not None else config_internal.default_window_size
        search_k = pre_k if pre_k is not None else (top_k * 5)

        retriever = self._build_ensemble_retriever(search_k=search_k, bm25_weight=bm25_weight, vector_weight=vector_weight)
        docs = retriever.invoke(question)

        raw_texts = [d.page_content for d in docs]
        raw_sources = [d.metadata.get('source_file') for d in docs]
        raw_ids = [d.metadata.get('child_id') for d in docs]

        sorted_texts, sorted_s, sorted_ids, sorted_scores = self._rerank_passages(question, raw_texts, raw_sources, raw_ids)

        final_contents, final_scores, final_sources, seen_hashes = [], [], [], set()

        import math
        def logit_to_prob(x):
            return 1 / (1 + math.exp(-max(min(x, 100), -100)))

        for c_id, score, src, c_text in zip(sorted_ids, sorted_scores, sorted_s, sorted_texts):
            if len(final_contents) >= top_k:
                break
            parent_text = self.parent_map.get(str(c_id).strip())
            content_to_use = self._get_smart_window(parent_text, c_text, w_size) if parent_text else c_text

            # 【修复 1】：Chunk 的 Logit 转为概率
            chunk_prob = logit_to_prob(score)

            # 句子级过滤逻辑
            sentences = nltk.tokenize.sent_tokenize(content_to_use)
            if not sentences or len(sentences) == 1:
                # 单句直接保留原 chunk
                if content_to_use.strip() not in seen_hashes:
                    seen_hashes.add(content_to_use.strip())
                    final_contents.append(content_to_use)
                    final_scores.append(chunk_prob)
                    final_sources.append(src)
                continue

            # 对每个句子用 cross-encoder 打分
            if self._cross_encoder is None:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(config_internal.rerank_model_name, device=config_internal.rerank_device)
            pairs = [[question, s[:1000]] for s in sentences]

            sent_scores = self._cross_encoder.predict(pairs).tolist()
            sent_probs = [logit_to_prob(sc) for sc in sent_scores]

            # Filter out low-probability sentences
            filtered = [(s, prob) for s, prob in zip(sentences, sent_probs) if prob >= config.sentence_filter_threshold]

            # Retention rule
            if filtered:
                max_sent_prob = max(prob for _, prob in filtered)
                # 【修复 3】：概率对比概率
                if chunk_prob >= max_sent_prob:
                    if content_to_use.strip() not in seen_hashes:
                        seen_hashes.add(content_to_use.strip())
                        final_contents.append(content_to_use)
                        final_scores.append(chunk_prob)
                        final_sources.append(src)
                else:
                    # MIGRES: Concatenate high-score sentences into a denoised passage
                    denoised_text = " ".join([s for s, _ in filtered]).strip()
                    if denoised_text and denoised_text not in seen_hashes:
                        seen_hashes.add(denoised_text)
                        final_contents.append(denoised_text)
                        final_scores.append(max_sent_prob)
                        final_sources.append(src)
            else:
                # If all sentences are below threshold, skip this chunk (strict denoising)
                continue

        return final_contents, final_scores, final_sources