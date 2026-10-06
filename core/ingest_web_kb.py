#!/usr/bin/env python3
"""
增量网络知识库向量化工具 (Web KB Ingestion Tool)
用于解析 archive_web_context_to_kb.py 生成的带 YAML Metadata 的 Markdown 文件，并将其插入底层的 ChromaDB 中。
"""

import os
import re
import yaml
import chromadb
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.rag_config import config

def parse_markdown_with_frontmatter(file_path: Path) -> Document:
    """提取 Markdown 中的 YAML Front Matter 作为 Langchain 向量库的 Metadata"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    metadata = {}
    # 匹配 YAML 格式
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if match:
        yaml_text = match.group(1)
        body = match.group(2).strip()
        try:
            metadata = yaml.safe_load(yaml_text) or {}
        except Exception as e:
            print(f"⚠️ 解析元数据失败 {file_path.name}: {e}")
    else:
        body = content.strip()
        
    # 保留关键溯源信息
    metadata['source_file'] = file_path.name
    return Document(page_content=body, metadata=metadata)

def ingest_archived_web_docs():
    base_dir = Path(__file__).resolve().parents[1] / "knowledge_base" / "raw_files" / "web_external" / "articles"
    if not base_dir.exists():
        print(f"❌ 目录不存在: {base_dir}")
        return

    markdown_files = list(base_dir.glob("*.md"))
    if not markdown_files:
        print("ℹ️ 没有发现需要入库的增量网络知识文件。")
        return

    print(f"📥 找到 {len(markdown_files)} 个网络扩展文档，正在提取元数据...")
    docs = [parse_markdown_with_frontmatter(f) for f in markdown_files]

    # 根据 rag_config 切分（保持与父子分块策略兼容）
    print("✂️ 正在进行文本切块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.child_chunk_size,
        chunk_overlap=config.child_overlap_size if hasattr(config, 'child_overlap_size') else 40
    )
    split_docs = text_splitter.split_documents(docs)
    
    # 获取与 advanced_retriever 中相同的 Chroma 实例
    chroma_persist_dir = Path(__file__).resolve().parents[1] / "knowledge_base" / "braincheck_vectordb"
    collection_name = "braincheck_advanced"
    
    print(f"🚀 连接到底层向量数据库 (Collection: {collection_name}) ...")
    client = chromadb.PersistentClient(path=str(chroma_persist_dir))
    
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        print("❌ 未能找到现有的集合，请确认是否已运行主文档的 ingest_documents.py 初始化库。")
        return

    # 这里使用了 HuggingFaceEmbeddings，你需要引入与 AdvancedRetriever 完全相同的嵌入模型
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=config.embed_model_name,
        encode_kwargs={"normalize_embeddings": getattr(config, 'embed_normalize', True)}
    )

    from langchain_community.vectorstores import Chroma
    vector_store = Chroma(
        client=client, 
        collection_name=collection_name, 
        embedding_function=embeddings
    )

    # 去重判定（基于源文件名），避免重复插入
    existing_sources = set(metadata.get('source_file') for metadata in collection.get()['metadatas'] if metadata)
    
    new_docs = [d for d in split_docs if d.metadata.get('source_file') not in existing_sources]
    
    if not new_docs:
        print("✅ 所有增量文件均已存在于数据库中，跳过入库。")
        return

    print(f"💾 正在将 {len(new_docs)} 个新块写入 ChromaDB 知识库...")
    vector_store.add_documents(new_docs)
    print("🎉 增量知识更新成功！闭环已闭合！")

if __name__ == "__main__":
    ingest_archived_web_docs()