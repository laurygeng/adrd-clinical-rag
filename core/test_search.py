#!/usr/bin/env python3
"""
Test script to verify if the newly archived web knowledge 
has been successfully ingested into ChromaDB.
"""

import os
import sys
from pathlib import Path

# Ensure the project root is in the python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from core.rag_config import config
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def main():
    chroma_path = PROJECT_ROOT / "knowledge_base" / "braincheck_vectordb"
    print(f"🚀 Connecting to ChromaDB at: {chroma_path}")
    
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection("braincheck_advanced")
        print(f"✅ Successfully connected to collection. Total chunks in DB: {collection.count()}")
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB: {e}")
        return
    
    print(f"🧠 Loading embedding model: {config.embed_model_name}...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=config.embed_model_name,
        encode_kwargs={"normalize_embeddings": getattr(config, 'embed_normalize', True)}
    )
    
    # We use a query heavily related to the missing knowledge we just grabbed
    # test_query = "What causes aggression in dementia patients? Are there specific medical conditions or communication difficulties?"
    # test_query = "What is the most common type of dementia? Does vascular dementia account for the majority of cases?"
    test_query = "Is delirium more prevalent in adults with or without underlying dementia? What is the relationship between delirium and dementia?"
    print(f"\n🔍 Searching for: '{test_query}'\n")
    
    # Execute vector search
    query_vector = embeddings.embed_query(test_query)
    results = collection.query(
        query_embeddings=[query_vector],
        # n_results=3  # Retrieve top 3 closest matches
        n_results=10  
    )
    
    # Print the results
    if not results['documents'][0]:
        print("⚠️ No results found.")
        return

    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"--- Match {i+1} ---")
        source_file = meta.get('source_file', 'Unknown')
        
        # Highlight if it's from our newly ingested web documents
        if source_file.startswith("web_"):
            print(f"📄 Source File: 🌟 {source_file} (NEW WEB KNOWLEDGE) 🌟")
        else:
            print(f"📄 Source File: {source_file} (LOCAL BASE KNOWLEDGE)")
            
        # Print the first 300 characters of the matched chunk
        print(f"📝 Content: {doc[:300]}...\n")

if __name__ == "__main__":
    main()