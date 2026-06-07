#!/usr/bin/env python3
import os
import json
import uuid
import pickle
import logging
from typing import List, Dict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from load_data import SimpleBrainCheckLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import chromadb
from rag_config import config

@dataclass
class ProcessedNode:
    child_id: str
    child_content: str
    parent_id: str
    parent_content: str
    metadata: Dict = field(default_factory=dict)

class IncrementalIngestionPipeline:
    def __init__(self):
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../knowledge_base/advanced_rag'))
        self.chroma_persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../knowledge_base/braincheck_vectordb'))
        os.makedirs(self.output_dir, exist_ok=True)

        input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../knowledge_base/raw_files'))
        self.loader = SimpleBrainCheckLoader(local_folder_path=input_dir)
        
        self.status_file = os.path.join(self.output_dir, "processed_files.json")
        self.processed_files = self._load_status()

        # Step 1: Structural splitter (Markdown-based) to keep lists and SOPs under the same header intact
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

        # Step 2: Token-based parent splitter to enforce max limits on large sections
        self.parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=config.parent_chunk_size, chunk_overlap=config.parent_chunk_overlap
        )
        
        # Step 3: Token-based child splitter for granular retrieval chunks
        self.child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=config.child_chunk_size, chunk_overlap=config.child_chunk_overlap
        )

    def _load_status(self) -> List[str]:
        if os.path.exists(self.status_file):
            with open(self.status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_status(self):
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(self.processed_files, f, ensure_ascii=False, indent=2)

    def run_ingestion(self):
        print("🔍 Scanning directory for new documents...")
        raw_docs = self.loader.load_documents()
        if not raw_docs:
            print("❌ No documents found in target folder.")
            return

        new_docs = [d for d in raw_docs if d.metadata.get('source_file') not in self.processed_files]
        if not new_docs:
            print("✨ All documents are up to date. No new files to ingest.")
            return

        print(f"🚀 Found {len(set(d.metadata.get('source_file') for d in new_docs))} new files. Processing...")
        new_docs = self.loader.preprocess_documents(new_docs)
        
        processed_nodes = []
        for doc in new_docs:
            # Apply structural markdown splitting first
            try:
                md_docs = self.markdown_splitter.split_text(doc.page_content)
                # Inherit original document metadata (like source_file) into the markdown-split docs
                for md_doc in md_docs:
                    merged_meta = doc.metadata.copy()
                    merged_meta.update(md_doc.metadata)
                    md_doc.metadata = merged_meta
            except Exception as e:
                logging.warning(f"Markdown splitting failed for a document, falling back to original doc: {e}")
                md_docs = [doc]

            # If the markdown splitter didn't yield anything, fallback to the original document
            if not md_docs:
                md_docs = [doc]

            # Apply token-based parent chunking to the structured sections
            parent_chunks = self.parent_splitter.split_documents(md_docs)
            
            for parent in parent_chunks:
                parent_id = str(uuid.uuid4())
                
                # Apply token-based child chunking
                child_chunks = self.child_splitter.split_documents([parent])
                for child in child_chunks:
                    meta = parent.metadata.copy()
                    meta['parent_id'] = parent_id
                    
                    # Optional: Weave the header hierarchy directly into the content for stronger semantic matching
                    header_context = " > ".join([v for k, v in meta.items() if k.startswith('Header')])
                    augmented_child_content = f"[{header_context}]\n{child.page_content}" if header_context else child.page_content

                    processed_nodes.append(
                        ProcessedNode(
                            child_id=str(uuid.uuid4()),
                            child_content=augmented_child_content,
                            parent_id=parent_id,
                            parent_content=parent.page_content,
                            metadata=meta
                        )
                    )
            self.processed_files.append(doc.metadata.get('source_file'))

        self._update_local_stores(processed_nodes)
        self._save_status()

    def _update_local_stores(self, nodes: List[ProcessedNode]):
        parent_map_path = os.path.join(self.output_dir, "parent_map.json")
        pkl_path = os.path.join(self.output_dir, "vector_ready_children.pkl")

        parent_map = {}
        if os.path.exists(parent_map_path):
            with open(parent_map_path, "r", encoding='utf-8') as f:
                parent_map = json.load(f)
        for node in nodes:
            parent_map[node.child_id] = node.parent_content
        with open(parent_map_path, "w", encoding='utf-8') as f:
            json.dump(parent_map, f, ensure_ascii=False, indent=2)

        vector_ready_docs = []
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                vector_ready_docs = pickle.load(f)
        
        new_vector_docs = []
        for node in nodes:
            # Add strict prefix to define content boundaries explicitly for the LLM
            augmented_content = f"Content: {node.child_content}"
            doc_meta = node.metadata.copy()
            doc_meta.update({"child_id": node.child_id, "parent_id": node.parent_id})
            doc = Document(page_content=augmented_content, metadata=doc_meta)
            
            vector_ready_docs.append(doc)
            new_vector_docs.append(doc)
            
        with open(pkl_path, "wb") as f:
            pickle.dump(vector_ready_docs, f)

        if new_vector_docs:
            print(f"🔗 Embedding and pushing {len(new_vector_docs)} new chunks to ChromaDB...")
            client = chromadb.PersistentClient(path=self.chroma_persist_dir)
            collection = client.get_or_create_collection("braincheck_advanced")
            
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(config.embed_model_name)
            
            texts = [d.page_content for d in new_vector_docs]
            embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()
            ids = [str(d.metadata.get('child_id')) for d in new_vector_docs]
            metadatas = [d.metadata for d in new_vector_docs]
            
            batch_size = config.chroma_batch_size
            for i in range(0, len(new_vector_docs), batch_size):
                end = min(i + batch_size, len(new_vector_docs))
                collection.add(
                    ids=ids[i:end], documents=texts[i:end], embeddings=embeddings[i:end], metadatas=metadatas[i:end]
                )
        print(f"✅ Successfully appended {len(nodes)} chunks to the RAG knowledge base.")

if __name__ == "__main__":
    pipeline = IncrementalIngestionPipeline()
    pipeline.run_ingestion()