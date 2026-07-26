# advanced_retriever.py
import os
import sys
import logging
import json
import pickle
import re
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
from typing import List, Tuple, Optional

# from rag_config import config
from core.rag_config import config

# Core dependency check
try:
    import chromadb
    from sentence_transformers import CrossEncoder
    from langchain_community.vectorstores import Chroma as LCChroma
    from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# BGE embeddings benefit from a query-side instruction + normalization, which
# HuggingFaceBgeEmbeddings handles natively. Fall back gracefully if unavailable.
try:
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
except Exception:
    HuggingFaceBgeEmbeddings = None

# BM25 & Ensemble compatibility. Import each SEPARATELY: previously both were in one try
# block, so when EnsembleRetriever's path changed (langchain 1.x moved it to
# langchain_classic), the single failure nulled BM25Retriever too — silently disabling the
# entire hybrid and leaving retrieval VECTOR-ONLY despite bm25_weight in config.
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    BM25Retriever = None
try:                                              # langchain >=1.0
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:                                          # older layouts
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        EnsembleRetriever = None

class AdvancedRetriever:
    def score_text(self, query: str, text: str) -> float:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(config.rerank_model_name, device=config.rerank_device)
        return float(self._cross_encoder.predict([[query, text[:config.rerank_max_chars]]])[0])
        
    def __init__(self):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.data_dir = os.path.join(os.path.abspath(os.path.join(current_script_dir, '../knowledge_base/advanced_rag')))
        self.chroma_persist_dir = os.path.join(os.path.abspath(os.path.join(current_script_dir, '../knowledge_base/braincheck_vectordb')))
        self.collection_name = "braincheck_advanced"

        self.children_pkl_path = os.path.join(self.data_dir, "vector_ready_children.pkl")
        self.parent_map_path = os.path.join(self.data_dir, "parent_map.json")

        self.chroma_collection = None
        self._cross_encoder = None
        
        self.bm25_retriever = None
        self.vector_retriever_base = None
        self.vector_store = None
        
        self.child_chunks = []
        self.parent_map = {}
        
        self._load_data()
        self._init_chroma()
        self._init_retrievers() 
        
    def _load_data(self):
        if os.path.exists(self.children_pkl_path):
            with open(self.children_pkl_path, 'rb') as f:
                self.child_chunks = pickle.load(f)
        else:
            raise FileNotFoundError(f"Missing {self.children_pkl_path}. Did you run ingest_documents.py?")
            
        if os.path.exists(self.parent_map_path):
            with open(self.parent_map_path, 'r', encoding='utf-8') as f:
                raw_map = json.load(f)
                self.parent_map = {str(k).strip(): v for k, v in raw_map.items()}
        else:
            logging.warning("⚠️ Parent map not found!")

    def _init_chroma(self):
        client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        try:
            self.chroma_collection = client.get_collection(self.collection_name)
            print(f"📂 Read-only mode: Successfully connected to ChromaDB. Current chunk count: {self.chroma_collection.count()}")
        except Exception:
            raise FileNotFoundError(f"❌ Chroma collection '{self.collection_name}' not found! Please run ingest_documents.py first.")

    def _build_query_embeddings(self):
        """Build the query-side embedding function.

        For BGE models we prepend the query instruction (passages are stored
        WITHOUT any instruction) and L2-normalize so cosine similarity is well
        defined. Falls back to plain HuggingFace embeddings otherwise.
        """
        normalize = getattr(config, "embed_normalize", True)
        instruction = getattr(config, "bge_query_instruction", "")
        if HuggingFaceBgeEmbeddings is not None and instruction:
            return HuggingFaceBgeEmbeddings(
                model_name=config.embed_model_name,
                query_instruction=instruction,
                embed_instruction="",
                encode_kwargs={"normalize_embeddings": normalize},
            )
        return SentenceTransformerEmbeddings(
            model_name=config.embed_model_name,
            encode_kwargs={"normalize_embeddings": normalize},
        )

    def _init_retrievers(self):
        print("🛠️ Preloading retrieval components...")
        if BM25Retriever and self.child_chunks:
            self.bm25_retriever = BM25Retriever.from_documents(self.child_chunks)

        hf_embeddings = self._build_query_embeddings()
        self.vector_store = LCChroma(
            persist_directory=self.chroma_persist_dir, 
            collection_name=self.collection_name, 
            embedding_function=hf_embeddings
        )
        print("✅ Retrieval components preloaded successfully.")

    def _get_smart_window(self, parent_text: str, child_text: str, window_size: int) -> str:
        match_target = child_text
        if "Content:" in match_target:
            match_target = match_target.split("Content:", 1)[-1]
        # Strip the injected header context prefix (e.g. "[Header 1 > Header 2]\n")
        # that ingestion prepends to child content. The parent text does NOT contain
        # this prefix, so leaving it in made every match fail and silently disabled
        # window expansion for any chunk that carried a header.
        match_target = re.sub(r'^\s*\[[^\]]*\]\s*', '', match_target).strip()
        if not match_target:
            return child_text

        # Whitespace-tolerant pattern: re.escape turns spaces/newlines into "\ " / "\\\n",
        # so collapse any run of (escaped) whitespace into a single \s+.
        query_pattern = re.sub(r'(\\?\s)+', r'\\s+', re.escape(match_target))
        match = re.search(query_pattern, parent_text)

        if not match: return child_text
            
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
        retrievers = []
        if self.bm25_retriever:
            self.bm25_retriever.k = search_k
            retrievers.append(self.bm25_retriever)
        
        retrievers.append(self.vector_store.as_retriever(search_kwargs={'k': search_k}))
        if len(retrievers) > 1:
            return EnsembleRetriever(retrievers=retrievers, weights=[bm25_weight, vector_weight])
        return retrievers[0]

    def _rerank_passages(self, query, passages, sources, ids):
        if not config.rerank_enabled: return passages, sources, ids, [1.0]*len(passages)
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(config.rerank_model_name, device=config.rerank_device)
        pairs = [[query, p[:config.rerank_max_chars]] for p in passages]
        scores = self._cross_encoder.predict(pairs).tolist()
        combined = sorted(zip(passages, sources, ids, scores), key=lambda x: x[3], reverse=True)
        return [x[0] for x in combined], [x[1] for x in combined], [x[2] for x in combined], [x[3] for x in combined]

    def get_retrieved_passages(self, question: str, top_k: int = 3, bm25_weight: float = 0.5, 
                               vector_weight: float = 0.5, pre_k: Optional[int] = None,
                               window_size: Optional[int] = None) -> Tuple[List[str], List[float], List[str]]:
        
        w_size = window_size if window_size is not None else config.retrieval_window_size
        search_k = pre_k if pre_k is not None else (top_k * 5)

        retriever = self._build_ensemble_retriever(search_k=search_k, bm25_weight=bm25_weight, vector_weight=vector_weight)
        docs = retriever.invoke(question)

        raw_texts = [d.page_content for d in docs]
        raw_sources = [d.metadata.get('source_file') for d in docs]
        raw_ids = [d.metadata.get('child_id') for d in docs]

        # Clean the candidate pool BEFORE reranking: drop academic-artifact noise (reference
        # lists, author/affiliation blocks, table/figure dumps, page markers) so the
        # cross-encoder spends its slots on real content instead of medium-score junk.
        # Non-destructive (vector DB untouched); reversible via config.kb_noise_filter.
        if getattr(config, "kb_noise_filter", True):
            from core.kb_noise import is_noise
            keep = [i for i, t in enumerate(raw_texts) if not is_noise(t)]
            if keep and len(keep) < len(raw_texts):   # never empty the pool
                raw_texts   = [raw_texts[i]   for i in keep]
                raw_sources = [raw_sources[i] for i in keep]
                raw_ids     = [raw_ids[i]     for i in keep]

        sorted_texts, sorted_s, sorted_ids, sorted_scores = self._rerank_passages(question, raw_texts, raw_sources, raw_ids)

        final_contents, final_scores, final_sources, seen_hashes = [], [], [], set()
        import math
        def logit_to_prob(x):
            return 1 / (1 + math.exp(-max(min(x, 100), -100)))

        for c_id, score, src, c_text in zip(sorted_ids, sorted_scores, sorted_s, sorted_texts):
            if len(final_contents) >= top_k:
                break
            chunk_prob = logit_to_prob(score)
            if chunk_prob >= config.rerank_min_prob:
                parent_text = self.parent_map.get(str(c_id).strip())
                content_to_use = self._get_smart_window(parent_text, c_text, w_size) if parent_text else c_text

                if content_to_use.strip() not in seen_hashes:
                    seen_hashes.add(content_to_use.strip())
                    final_contents.append(content_to_use)
                    final_scores.append(chunk_prob)
                    final_sources.append(src)

        return final_contents, final_scores, final_sources
    
    
    def rerank_texts(self, query: str, passages: List[str], sources: List[str]):
        """
        Rerank arbitrary passages (e.g., web passages) using the same CrossEncoder.
        Returns (sorted_passages, sorted_sources, sorted_scores_logits)
        """
        if not passages:
            return [], [], []
        dummy_ids = [str(i) for i in range(len(passages))]
        sorted_texts, sorted_s, _, sorted_scores = self._rerank_passages(query, passages, sources, dummy_ids)
        return sorted_texts, sorted_s, sorted_scores
