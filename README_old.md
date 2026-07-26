# ADRD Clinical Knowledge Retrieval Framework (Advanced RAG Pipeline)

An advanced, enterprise-grade Retrieval-Augmented Generation (RAG) framework designed for optimizing clinical support and evidence-based fact-checking within the Alzheimer's Disease and Related Dementias (ADRD) domain. The pipeline integrates hierarchical parent-child chunking, hybrid ensemble retrieval, Cross-Encoder reranking, semantic window recovery, and LLM-driven query rewriting and validation.

---

## 📂 File Architecture & Component Breakdown

### 1. `rag_config.py`
* **Purpose**: Centralized configuration management for the entire RAG pipeline ecosystem.
* **Key Functionalities**:
  * Definining tuning parameters for hierarchical text splitting (e.g., `parent_chunk_size=800`, `child_chunk_size=250`).
  * Storing weights for retrieval ensembles (`bm25_weight=0.5`, `vector_weight=0.5`) and setting candidate thresholds (`pre_k`, `top_k`).
  * Specifying local embedding models (`all-MiniLM-L6-v2`), local re-ranking models (`cross-encoder/ms-marco-MiniLM-L-6-v2`), and remote evaluation engines (`gpt-4o`).
  * Automatically handles device mapping (`cuda` vs `cpu`) for optimal deep learning performance.

### 2. `load_data.py`
* **Purpose**: Robust document extraction, text pre-cleaning, and qualitative testing utility.
* **Key Functionalities**:
  * Orchestrates text extraction across varying file extensions (`.pdf`, `.docx`, `.txt`) using optimized loaders.
  * Implements rule-based heuristics to remove structural layout noise (headers, footers, citation pages, and URL blocks).
  * Feature an advanced spacing normalization regex engine (`_normalize_pdf_spacing`) to resolve common OCR/PDF layout artifacts like letter-spaced words or broken line-break hyphenations.
  * Provides random sampling and keyword-based qualitative checks to monitor text split integrity before pushing to vector databases.

### 3. `ingest_documents.py`
* **Purpose**: Incremental document processing pipeline and database manager.
* **Key Functionalities**:
  * Tracks already-processed files via an incremental state ledger (`processed_files.json`) to skip unnecessary processing.
  * Executes a hierarchical layout split: document texts are chunked into broad "Parent Nodes", which are then subdivided into high-density "Child Nodes".
  * Maps every Child ID back to its parent context string inside `parent_map.json`.
  * Generates local vector embeddings for child nodes and pushes them along with comprehensive metadata blocks into a unified ChromaDB collection (`braincheck_advanced`).

### 4. `advanced_retriever.py`
* **Purpose**: Multi-stage hybrid search engine and context-window builder.
* **Key Functionalities**:
  * Builds a runtime ensemble retriever pairing lexical keyword indexes (BM25) with semantic dense vectors (Chroma).
  * Executes neural re-ranking on candidate chunks via a localized Cross-Encoder model to compute high-precision alignment scores.
  * Employs a dynamic window recovery algorithm (`_get_smart_window`): locating the retrieved child snippet inside its overarching parent text and gracefully extending boundaries to the closest complete sentence edge to preserve surrounding contextual context.

### 5. `llm_utils.py`
* **Purpose**: LLM interaction wrapper handling structured prompting, optimization, and fact verification.
* **Key Functionalities**:
  * `rewrite_tf_query`: Decomposes complex True/False assertions into distinct interrogative queries for targeted retrieval.
  * `decompose_mc_options`: Parses multiple-choice questions independently to formulate key factual keyword search queries for every candidate option.
  * `evaluate_tf_evidence`: Evaluates statements strictly against retrieved contexts using advanced Chain-of-Thought (CoT) prompts to deliver verdicts (`True`, `False`, `insufficient`).
  * `evaluate_context`: Gauges overall context adequacy and explicitly logs missing elements if an item is flagged as `unanswerable`.

### 6. `run_retrieval_adrd.py`
* **Purpose**: Main execution orchestrator running evaluation workloads against the ADRD Caregiving benchmark dataset.
* **Key Functionalities**:
  * Ingests evaluation benchmarks (`ADRD_Caregiving_Multiple_Choice.json` & `ADRD_Caregiving_True_or_False.json`).
  * Loops over validation sets, utilizing `llm_utils` for problem decomposition, executing hybrid queries via `AdvancedRetriever`, and validating downstream answerability.
  * Computes final results and serializes analytical output tables containing retrieved text spans, source filenames, answerability statuses, and factual judgments into a clean CSV format.

---

## 📈 Standard Pipeline Execution Order

To run the pipeline smoothly from raw document ingestion to comprehensive evaluation, execute the scripts sequentially using the following layout order:

### Prerequisites: Environment Initialization
Before initiating any scripts, ensure your system has your API credentials loaded to authorize the validation agents. Navigate to your project directory and export your keys:
```bash
export OPENAI_API_KEY="your-secured-openai-api-key-here"
# Optional: export OPENAI_BASE_URL="your-custom-proxy-endpoint"