#!/usr/bin/env python3
import os
import json
import uuid
import pickle
import logging
import networkx as nx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from load_data import SimpleBrainCheckLoader
except ImportError:
    raise ImportError("Could not import SimpleBrainCheckLoader from load_data.py")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

@dataclass
class ProcessedNode:
    """Stores information for a processed parent-child node pair."""
    child_id: str
    child_content: str
    parent_id: str
    parent_content: str
    generated_questions: List[str] = field(default_factory=list)
    entities: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class AdvancedRAGProcessor:
    def __init__(self, output_dir=None):
        # 输出目录固定为 code/knowledge_base/advanced_rag
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../knowledge_base/advanced_rag'))
        os.makedirs(self.output_dir, exist_ok=True)

        # 输入目录固定为 code/knowledge_base/raw_files
        input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../knowledge_base/raw_files'))
        self.loader = SimpleBrainCheckLoader(local_folder_path=input_dir)
        
        # 1. LLM Client Check
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error("❌ OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL")
        )
        self.model_name = os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini")

        # 2. 🟢 参数调整：缩减粒度以提升 Context Relevance
        print("✂️  Initializing Optimized Token-Aware Splitters...")
        
        # Parent Splitter: 从 2000 降至 800 tokens，减少背景噪声
        self.parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=800,
            chunk_overlap=100
        )
        
        # Child Splitter: 从 400 降至 250 tokens，提供更精准的语义定位
        self.child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=250,
            chunk_overlap=40
        )
        
        self.knowledge_graph = nx.Graph()

    def generate_synthetic_qa(self, text: str) -> List[str]:
        """Generates 2 specific questions that this text answers."""
        prompt = f"""
        Analyze the following text snippet and generate 2 specific questions that this text answers. 
        Output ONLY the questions, separated by a newline. No numbering.
        
        Text: "{text[:600]}..."
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = response.choices[0].message.content
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            return questions[:2]
        except Exception:
            return []

    def extract_graph_triples(self, text: str) -> List[Dict]:
        """Extracts key entities and their relationships (Head -> Relation -> Tail)."""
        prompt = f"""
        Extract top 3 key entities and their relationships from the text. 
        Format as JSON list: [{{"head": "Entity A", "relation": "relationship", "tail": "Entity B"}}]
        Return ONLY JSON.
        
        Text: "{text[:600]}..."
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical knowledge graph extractor. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            # Standardize output format
            if isinstance(data, dict):
                return data.get('triples', data.get('relationships', data.get('entities', [])))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _process_single_child(self, child_doc, parent_doc, doc_meta):
        """Helper function to process a single child chunk."""
        child_id = str(uuid.uuid4())
        questions = self.generate_synthetic_qa(child_doc.page_content)
        triples = self.extract_graph_triples(child_doc.page_content)
        
        return ProcessedNode(
            child_id=child_id,
            child_content=child_doc.page_content,
            parent_id=doc_meta.get('parent_id'),
            parent_content=parent_doc.page_content,
            generated_questions=questions,
            entities=triples,
            metadata=doc_meta
        )

    def process_documents(self):
        """Main execution flow."""
        print(f"🚀 Starting Advanced RAG Processing (Model: {self.model_name})...")
        raw_docs = self.loader.load_documents()
        if not raw_docs:
            print("❌ No documents found.")
            return

        raw_docs = self.loader.preprocess_documents(raw_docs)
        processed_nodes = []
        
        # Prepare tasks for concurrency
        tasks = []
        # Max workers 10 for faster LLM processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            for doc in raw_docs:
                parent_chunks = self.parent_splitter.split_documents([doc])
                for parent in parent_chunks:
                    parent_id = str(uuid.uuid4())
                    child_chunks = self.child_splitter.split_documents([parent])
                    for child in child_chunks:
                        meta = doc.metadata.copy()
                        meta['parent_id'] = parent_id
                        tasks.append(executor.submit(self._process_single_child, child, parent, meta))
            
            for future in tqdm(as_completed(tasks), total=len(tasks), desc="🧠 Enriching Chunks"):
                try:
                    node = future.result()
                    processed_nodes.append(node)
                    # Sync Graph Data
                    for triple in node.entities:
                        h, r, t = triple.get('head'), triple.get('relation'), triple.get('tail')
                        if h and r and t:
                            self.knowledge_graph.add_edge(h, t, relation=r, source_child_id=node.child_id)
                except Exception as e:
                    logging.error(f"Error processing chunk: {e}")

        self._save_results(processed_nodes)

    def _save_results(self, nodes: List[ProcessedNode]):
        """Saves structural data for the Retriever."""
        parent_map = {}
        vector_ready_docs = [] 
        
        for node in nodes:
            parent_map[node.child_id] = node.parent_content
            
            # 3. 🟢 结构化增强：加入 Topic Tags 方便 CRAG 判定
            entities_str = ", ".join(list(set([t.get('head', '') for t in node.entities])))
            augmented_content = (
                f"Topic Tags: {entities_str}\n"
                f"Questions: {' '.join(node.generated_questions)}\n\n"
                f"Content: {node.child_content}"
            )
            
            doc_meta = node.metadata.copy()
            doc_meta.update({
                "child_id": node.child_id,
                "parent_id": node.parent_id,
                "has_graph_data": len(node.entities) > 0,
                "content_len": len(node.child_content)
            })
            vector_ready_docs.append(Document(page_content=augmented_content, metadata=doc_meta))

        # Save files
        with open(os.path.join(self.output_dir, "parent_map.json"), "w", encoding='utf-8') as f:
            json.dump(parent_map, f, ensure_ascii=False, indent=2)
            
        nx.write_gml(self.knowledge_graph, os.path.join(self.output_dir, "knowledge_graph.gml"))
        
        with open(os.path.join(self.output_dir, "vector_ready_children.pkl"), "wb") as f:
            pickle.dump(vector_ready_docs, f)

        print(f"\n✅ Processing Complete! Parent size reduced, logic enriched.")

if __name__ == "__main__":
    processor = AdvancedRAGProcessor()
    processor.process_documents()