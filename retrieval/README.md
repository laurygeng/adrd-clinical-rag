# 检索模块说明（code/retrieval/）

本目录包含 ADRD 项目的核心检索流程及相关脚本。以下为各脚本功能、检索流程执行顺序、主要产物及其用途说明。


## 1. 各脚本作用

- **load_data.py**
  - 原始文档加载、清洗、分块、关键词索引、知识库构建与持久化（pkl/embedding/向量库）。
  - 主要产物：
    - `knowledge_base/braincheck_knowledge_base.pkl`：分块后的知识库（无embedding）。
    - `knowledge_base/braincheck_knowledge_base_with_emb.pkl`：带embedding的知识库。
    - `knowledge_base/braincheck_vectordb/`：Chroma向量数据库（用于后续检索）。
  - 执行顺序：
    1. 加载 raw_files 下所有原始文档。
    2. 文本清洗、分块（token/char递归分割）。
    3. 构建关键词索引。
    4. 保存知识库（pkl），生成embedding（可选），持久化到Chroma向量库。
    5. 产物供 generate_parent_child_chunks.py、检索流程等后续使用。

- **generate_parent_child_chunks.py**
  - 基于知识库分块，抽取父子结构、生成知识三元组、问题生成、结构化数据产出，供高级检索用。
  - 主要产物（均在 `knowledge_base/advanced_rag/`）：
    - `vector_ready_children.pkl`：结构化、增强后的子块（含profile、标签、问题等）。
    - `parent_map.json`：父子结构映射，支持窗口扩展检索。
    - `knowledge_graph.gml`：知识三元组图谱（可视化、关系分析）。
  - 执行顺序：
    1. 加载知识库分块。
    2. 并行处理每个块，抽取父子关系、实体、生成问题、三元组。
    3. 产出结构化数据文件，供 advanced_retriever.py 检索流程直接使用。

- **run_retrieval_adrd.py**
  - 批量运行 ADRD-Bench 检索流程，支持多项选择题与判断题。
  - 通过命令行参数设置检索参数（top_k、window、bm25、vector 等），调用 AdvancedRetriever 完成检索、融合、重排序。
  - 产出标准化的检索结果 CSV 文件。

- **advanced_retriever.py**
  - 检索主引擎，集成 BM25、向量检索、父子结构窗口扩展、cross-encoder 重排序等。
  - 负责加载结构化知识库、初始化向量数据库（Chroma）、构建检索器、执行检索与重排序。
  - 被 run_retrieval_adrd.py 直接调用。

- **rag_config.py**
  - 集中管理检索流程的核心参数（top_k、window、bm25、vector 等），便于统一配置和实验复现。
  - 被 run_retrieval_adrd.py 引用。



## 2. 检索流程执行顺序

1. **第一步：构建基础知识库**
  - 运行 `load_data.py`，处理原始文档，生成知识库分块、embedding 及向量数据库。
  - 产物：`knowledge_base/braincheck_knowledge_base.pkl`、`knowledge_base/braincheck_knowledge_base_with_emb.pkl`、`knowledge_base/braincheck_vectordb/`。

2. **第二步：生成父子结构化数据**
  - 运行 `generate_parent_child_chunks.py`，基于知识库分块生成父子结构、三元组、增强子块等结构化产物。
  - 产物：`knowledge_base/advanced_rag/vector_ready_children.pkl`、`parent_map.json`、`knowledge_graph.gml`。

3. **第三步：批量检索**
  - 运行 `run_retrieval_adrd.py`，批量读取问题，调用高级检索器。
  - 依赖：前两步生成的所有结构化产物。
  - 运行时会自动调用 `rag_config.py`（统一参数配置）和 `advanced_retriever.py`（检索主引擎）。
  - 产物：标准化检索结果 CSV 文件（如 `retrieval_results/` 下的检索结果）。

> **注意：**
> - 必须严格按照上述顺序依次执行，确保所有依赖文件已生成，否则检索流程无法正常运行。
> - 检索参数可在 `rag_config.py` 统一配置，也可通过命令行参数覆盖。


## 3. 主要产物及用途

- **检索结果 CSV**（如 `retrieval_ADRD_all_k3_w500_20260526_153434.csv`）
  - 位置：`code/retrieval/retrieval_results/` 或指定输出目录
  - 字段：Question_ID, Type, Question, Retrieved_Passages, Retrieved_Sources, Rerank_Scores
  - 用途：为后续大模型生成答案、评测等流程提供标准化检索上下文。

- **结构化知识库文件**（依赖项，非本流程产出）
  - `code/knowledge_base/advanced_rag/vector_ready_children.pkl`：分块知识内容，供检索用。
  - `code/knowledge_base/advanced_rag/parent_map.json`：父子结构映射，支持窗口扩展。


## 4. 典型流程总结

1. 确保结构化知识库已生成。
2. 运行 `run_retrieval_adrd.py`，产出检索结果 CSV。
3. 检索结果可直接用于答案生成、评测等下游任务。

---
如需调整检索策略，仅需修改 rag_config.py 或命令行参数，无需改动主流程代码。
