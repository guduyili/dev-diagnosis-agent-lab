"""D08 / W03-B：从当前文件的数据出发，真实分块、入库、工具检索和 parent 回取。

运行：python -m pytest tests/daily/test_d08_retrieval.py -v
自己改：在 DOCUMENTS 和参数表一起新增来源、查询及期望片段；不要只增加恒定答案。
"""
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector

import config
import db.vector_db_manager as vector_module
import rag_agent.tools as tools_module
from document_chunker import DocumentChunker
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager
from rag_agent.tools import ToolFactory

pytestmark = pytest.mark.day(8)

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


@pytest.mark.parametrize("query,expected_source,expected_text", [
    ("response_format", "deepseek.md", "function_calling"),
    ("parent_id", "parents.md", "page_content"),
    ("thread_id", "sessions.md", "InMemorySaver"),
])
def test_actual_search_returns_expected_source_and_parent(tmp_path, monkeypatch, query, expected_source, expected_text):
    # 1. 数据在本文件；每个参数用例独立创建知识库，没有“先跑某天”的要求。
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    chunker = DocumentChunker()
    parent_store = ParentStoreManager(tmp_path / "parents")
    # ToolFactory 源码内部会调用无参 ParentStoreManager，这里显式指定真实临时 store。
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: parent_store)
    vector_db = VectorDbManager()
    try:
        vector_db.create_collection(config.CHILD_COLLECTION)
        collection = vector_db.get_collection(config.CHILD_COLLECTION)
        for name, text in DOCUMENTS.items():
            path = tmp_path / name
            path.write_text(text, encoding="utf-8")
            parents, children = chunker.create_chunks_single(path, source_name=name)
            parent_store.save_many(parents)
            collection.add_documents(children)

        # 2. 直接创建上游工具，实际 invoke 得到文本。
        factory = ToolFactory(collection)
        tools = {tool.name: tool for tool in factory.create_tools()}
        result = tools["search_child_chunks"].invoke({"query": query, "limit": 1})

        # 3. 使用实际返回的 ID 回取，不预先把期望 parent 当作召回结果。
        assert f"File Name: {expected_source}" in result
        parent_id = result.split("Parent ID: ", 1)[1].splitlines()[0]
        parent_text = tools["retrieve_parent_chunks"].invoke({"parent_id": parent_id})
        assert expected_text in parent_text
        assert parent_store.load_content(parent_id)["content"] in parent_text
    finally:
        vector_db._VectorDbManager__client.close()


def test_search_propagates_query_k_and_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    monkeypatch.setattr(config, "RETRIEVAL_SCORE_THRESHOLD", 0.4)
    parent_store = ParentStoreManager(tmp_path / "parents")
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: parent_store)
    chunker = DocumentChunker()
    vector_db = VectorDbManager()
    try:
        vector_db.create_collection(config.CHILD_COLLECTION)
        collection = vector_db.get_collection(config.CHILD_COLLECTION)
        for name, text in DOCUMENTS.items():
            path = tmp_path / name
            path.write_text(text, encoding="utf-8")
            parents, children = chunker.create_chunks_single(path, source_name=name)
            parent_store.save_many(parents)
            collection.add_documents(children)

        # wraps 仅记录参数，仍执行原来的 similarity_search。
        spy = Mock(wraps=collection.similarity_search)
        monkeypatch.setattr(collection, "similarity_search", spy)
        factory = ToolFactory(collection)
        result = factory._search_child_chunks("parent_id", limit=2)

        spy.assert_called_once_with("parent_id", k=2, score_threshold=0.4)
        assert result.count("Parent ID: ") == 2
    finally:
        vector_db._VectorDbManager__client.close()
