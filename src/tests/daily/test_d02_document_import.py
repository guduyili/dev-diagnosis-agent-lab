"""D04 / W02-A：从 Markdown 到 parent JSON 和 child 向量的真实导入。

运行：python -m pytest tests/daily/test_d04_document_import.py -v
每个测试完整展示自己的准备过程；SimpleNamespace 只装入真实组件，不实现业务逻辑。
自己改：换文件内容、增加同名文件，观察“added/skipped”与实际存储之间的关系。
"""
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector

import config
import db.vector_db_manager as vector_module
from document_chunker import DocumentChunker
from core.document_manager import DocumentManager
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager

pytestmark = pytest.mark.day(2)




class LocalDenseEmbeddings(Embeddings):
    """只替代 embedding 模型；固定词表方便手算，不代表真实语义质量。"""
    def embed_query(self, text):
        terms = ("response_format", "parent_id", "thread_id", "pytest")
        return [float(text.lower().count(term)) for term in terms] + [0.01]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class LocalSparseEmbeddings(SparseEmbeddings):
    """固定稀疏向量；真实 Qdrant 负责写入、查询和融合。"""
    def embed_query(self, text):
        values = LocalDenseEmbeddings().embed_query(text)
        indices = [i for i, value in enumerate(values) if value > 0]
        return SparseVector(indices=indices, values=[values[i] for i in indices])

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]



# 以下学习资料摘写自 nodes.py、document_chunker.py、parent_store_manager.py、graph.py。
# 400 错误文字来自用户报告；这些短资料不计入生产问题数据集。
DOCUMENTS = {
    "deepseek.md": (
        "# response_format 错误记录\n\n"
        "用户报告的实际错误文本是：This response_format type is unavailable now。\n"
        '本地 rewrite_query 使用 with_structured_output(QueryAnalysis, method="function_calling")。\n'
        "它将结构化结果放进工具参数，避免依赖 json_schema 响应格式。\n"
        "离线构造成功不能证明 DeepSeek 在线请求已经成功。\n"
    ),
    "parents.md": (
        "# parent_id 与父块存储\n\n"
        "DocumentChunker 使用文件 stem 和块序号生成 parent_id，例如 parents_p0。\n"
        "child 的 metadata 包含 parent_id 和 source。\n"
        "ParentStoreManager 将 page_content 和 metadata 保存为 JSON。\n"
        "相同 parent_id 再次保存会覆盖原文件；它不是内容哈希。\n"
    ),
    "sessions.md": (
        "# thread_id 与检查点\n\n"
        "create_agent_graph 使用 InMemorySaver 保存进程内状态。\n"
        "同一个 thread_id 用于查询、更新和恢复同一检查点。\n"
        "不同 thread_id 可保存不同的图状态。\n"
        "Gradio 当前共享一个 RAGSystem，不能因此声称页面已自动隔离不同用户。\n"
    ),
}