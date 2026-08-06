#!/usr/bin/env python3
"""
Ingest authoritative PDFs/text into the local Chroma vector store used by AdvancedRetriever.

Usage:
  python code/tools/ingest_authoritative_pdfs.py --input_dir /path/to/pdfs --vector_dir /path/to/chroma --collection braincheck_advanced

If you omit --vector_dir/--collection the script will try to read defaults from AdvancedRetriever.
"""
import os
import sys
import argparse

# Allow importing AdvancedRetriever when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'retrieval')))
try:
    from advanced_retriever import AdvancedRetriever
except Exception:
    AdvancedRetriever = None

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    # Prefer the newer packaged adapter when available
    from langchain_chroma import Chroma as LCChroma
    CHROMA_NEEDS_PERSIST = False
except Exception:
    from langchain_community.vectorstores import Chroma as LCChroma
    CHROMA_NEEDS_PERSIST = True

from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings


def expand_local_knowledge_base(input_dir, vector_store_dir=None, collection_name=None, embedding_model_name='all-MiniLM-L6-v2'):
    # If possible, get defaults from AdvancedRetriever
    if not vector_store_dir or not collection_name:
        if AdvancedRetriever is not None:
            ar = AdvancedRetriever()
            vector_store_dir = vector_store_dir or ar.chroma_persist_dir
            collection_name = collection_name or ar.collection_name
        else:
            raise ValueError("Vector store directory or collection not provided and AdvancedRetriever unavailable.")

    print(f"📂 Input dir: {input_dir}")
    print(f"📦 Vector store: {vector_store_dir} | collection: {collection_name}")

    # Prepare embedding function
    hf_embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # If AdvancedRetriever available, check embedding model alignment
    try:
        if AdvancedRetriever is not None:
            # The project's AdvancedRetriever uses 'all-MiniLM-L6-v2' by default.
            project_default = 'all-MiniLM-L6-v2'
            if embedding_model_name != project_default:
                print(f"⚠️ Embedding model mismatch: this script uses '{embedding_model_name}' but AdvancedRetriever expects '{project_default}'.")
                print("   如果向量数据库中已有向量，请确认模型维度一致以避免维度不匹配错误。")
    except Exception:
        pass

    # Initialize Chroma wrapper
    db = LCChroma(persist_directory=vector_store_dir, collection_name=collection_name, embedding_function=hf_embeddings)

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    # Gather files
    files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.lower().endswith(('.pdf', '.txt'))]
    if not files:
        print("⚠️ No PDF or TXT files found in input directory.")
        return

    # Simple safeguard to avoid duplicate ingestion: track ingested filenames
    ingested_log = os.path.join(vector_store_dir, "ingested_files.txt")
    existing_ingested = set()
    try:
        if os.path.exists(ingested_log):
            with open(ingested_log, 'r', encoding='utf-8') as f:
                existing_ingested = set(l.strip() for l in f if l.strip())
    except Exception:
        existing_ingested = set()

    total_chunks = 0
    for fp in files:
        base = os.path.basename(fp)
        if base in existing_ingested:
            print(f"⏭️ 已跳过已入库文件: {base}")
            continue
        print(f"📄 Loading: {os.path.basename(fp)}")
        try:
            if fp.lower().endswith('.pdf'):
                loader = PyPDFLoader(fp)
            else:
                loader = TextLoader(fp, encoding='utf-8')
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            # Ensure metadata includes source_file
            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata['source_file'] = os.path.basename(fp)

            if chunks:
                db.add_documents(chunks)
                total_chunks += len(chunks)
                print(f"   ✂️  Added {len(chunks)} chunks from {os.path.basename(fp)}")
                # mark file as ingested
                try:
                    with open(ingested_log, 'a', encoding='utf-8') as f:
                        f.write(base + '\n')
                    existing_ingested.add(base)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Failed to ingest {fp}: {e}")
    # Persist only when using older langchain_community adapter that requires it
    try:
        if CHROMA_NEEDS_PERSIST and hasattr(db, 'persist'):
            db.persist()
    except Exception as e:
        print(f"⚠️ Persist failed or not needed: {e}")
    print(f"✅ Ingestion complete. Total chunks added: {total_chunks}")


def main():
    parser = argparse.ArgumentParser(description="Ingest authoritative PDFs/text into local Chroma vector DB")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing PDFs or text files')
    parser.add_argument('--vector_dir', type=str, default=None, help='Chroma persist directory (optional)')
    parser.add_argument('--collection', type=str, default=None, help='Chroma collection name (optional)')
    parser.add_argument('--embedding', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    args = parser.parse_args()

    expand_local_knowledge_base(
        input_dir=args.input_dir,
        vector_store_dir=args.vector_dir,
        collection_name=args.collection,
        embedding_model_name=args.embedding,
    )


if __name__ == '__main__':
    main()
