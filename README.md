### 一、 离线构建与准备阶段的文件（建库、清洗、切片）
结论：这些文件只在最初搭建知识库或增量更新文献时运行一次，不需要参与每次问答的 RAG 主流程。

ingest_documents.py（代码片段 16）
作用：全量/增量知识库构建流水线。负责扫描原始 PDF/Word、使用 Markdown 结构化切片、生成父子块映射（parent_map.json）、计算向量并写入 ChromaDB 数据库。  
离线建库脚本。只有当你添加了新的医学文献 PDF、需要重新建库时才运行它。

detect_visual_pdfs.py / find_teepa.py（代码片段 8、12）
作用：PDF 诊断与文件归类工具。用来扫描哪些 PDF 是扫描版/图片版需要走 VLM 处理，或者把散落的文件移动到指定目录。 数据预处理辅助工具。建库前运行一次即可。

load_data.py（代码片段 20）
作用：BrainCheck 基础文档加载器，负责文本清洗、PDF 文本提取和基础分块。  
：被 ingest_documents.py 内部调用的底层库文件。主流程不直接调用它，它是 ingest 的一部分。

二、 算法验证与离线评测阶段的文件（实验、AUC 评测）
结论：这些文件是你在做学术研究、调参对比、验证不同 Gate（门控机制）效果时写的评测脚本，不参与实际线上 RAG 问答。


一、 各代码片段的作用与组件分析

### Blackboard（黑板对象 —— 代码片段 2）
作用：充当整个系统的共享工作内存（Working Memory）。所有代理（Agent）都通过读写这个单一的 dataclass 对象来交换数据。  特点：数据结构确定、透明，非常方便用于审计、追踪（Trace）和消融实验（Ablation Logging）。  

### Adversarial Web-Research Agent（对抗性网络检索代理 —— 代码片段 1）
作用：自动识别缺失信息并自动补全。  特点：采用“双重人格（Dual Personality）”策略，针对一个陈述同时触发 支持性（Support） 和 反驳性（Refute） 两条检索路径。集成了 Exa（通用/政策）、Europe PMC（结构化医学摘要）、Tavily 等多个免费/付费后端，能够把网络上的最新或补充证据抓取下来。 

 ### Mesh Ontology & Entity Gate（MeSH 本体边界网关 —— 代码片段 3）
作用：对检索到的信息进行确定性的语义/本体校验。  特点：利用美国国立医学图书馆的 MeSH 概念层级（Sparql/Lookup API），判断检索证据和原始声明之间是否属于“概念泄漏”（例如：不能把广泛类群“痴呆”的性质直接套用到特定亚型“阿尔茨海默病病理”上）。这是一种高精度的硬编码规则边界。  

### Verification Court（异构验证法庭 —— 代码片段 5）
作用：对补全的证据进行多维度交叉校验（Judge Panel）。  特点：由三个专家级Judge组成：Entity Judge：结合了 MeSH 本体网关与 LLM 回退，防止概念层级错乱。  Modal Judge：检查语气/程度（Modal/Degree）是否被悄悄降级（例如把“必须/强制”篡改成“可能/有助于”）。  Fact Judge：三票自洽性投票（Self-consistency），判断整体文献是否支持该陈述。  Arbiter（仲裁员/Veto机制）：一票否决制。如果实体不匹配或语气严重降级，直接判定证据不足（INSUFFICIENT）。 

### Orchestrator Agent（编排调度器 —— 代码片段 4）
作用：控制执行顺序与流程。  
特点：不依赖复杂的框架（如 LangGraph），而是通过一个透明的 Python 控制流循环，将答案生成、网络检索和法庭校验串联起来。  


### 0805 log更新
rag_config.py：这是系统的全局配置文件，advanced_retriever.py 强依赖它来读取切片大小、检索数量（top_k）、使用的模型名称等核心参数。  trace_logger.py：这是追踪和日志记录工具，critic_agent.py 和 orchestrator.py 都在大量调用它来生成 .md 日志文件和 .jsonl 数据记录。  kb_noise.py：这是知识库清理组件。在 advanced_retriever.py 的重排（Rerank）阶段之前，系统会调用它过滤掉论文参考文献、作者信息等噪声文本，以保证传给模型的上下文足够纯净。  


这两个文件是之前为了维护复杂的分支逻辑而存在的，现在已经完全被架空：
verification_agent.py
mesh_ontology.py

orchestrator.py 中，已经将所有题型的评估交给了统一的 Identify-then-Verify (ItV) 机制。
原本由 verification_agent.py 负责的“法庭审核（Court Auditing）”步骤已经在代码中被硬编码为跳过