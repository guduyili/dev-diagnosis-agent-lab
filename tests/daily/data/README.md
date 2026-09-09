# 测试数据来源

这些短资料整理自本地上游实现及已报告的 400 错误，不是生产事故数据集。

- deepseek.md：用户报告的错误文本，以及 project/rag_agent/nodes.py 的 rewrite_query。
- parents.md：project/document_chunker.py、project/db/parent_store_manager.py。
- sessions.md：project/rag_agent/graph.py、project/core/rag_system.py、project/ui/gradio_app.py。
- retrieval_cases.json：以上资料的 query、期望来源文件和原文片段标注。

使用固定词表 embedding，验证真实存储、查询和证据连接，不代表 Qwen/BM25 语义效果。
资料经真实 DocumentManager 写入 pytest 临时目录中的 JSON/Qdrant，不修改现有知识库。
空文件、超长文本和坏 JSON 是对应测试中明确构造的边界输入，不计入真实业务案例。
