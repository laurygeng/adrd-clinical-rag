#!/usr/bin/env python3
import os

# Disable Hugging Face tokenizers parallelism
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Minimal config for retrieval parameters only
class RAGConfig:
    def __init__(self):
        self.retrieval_top_k = 3
        self.retrieval_pre_k = None
        self.retrieval_window = 500
        self.retrieval_bm25_weight = 0.5
        self.retrieval_vector_weight = 0.5
        # Maximum number of iterations for multi-round retrieval
        self.max_iterations = 3
        # Sentence-level filter threshold (sentences below this score will be filtered out)
        self.sentence_filter_threshold = 0.01

config = RAGConfig()