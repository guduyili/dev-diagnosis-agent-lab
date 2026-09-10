"""D07 / W03-A：真实 parent JSON 与 Qdrant 的存储行为。

运行：python -m pytest tests/daily/test_d07_stores.py -v
自己改：增加坏 JSON、不同文件同序号、同维度不同模型等反例。
前三步都在当前函数：准备数据 -> 创建真实管理器 -> 调用与断言。
"""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector
from qdrant_client.http import models

import config
import db.vector_db_manager as vector_module
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager

pytestmark = pytest.mark.day(7)

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


def test_parent_store_round_trip_overwrite_and_numeric_order(tmp_path):
    store = ParentStoreManager(tmp_path / "parents")

    store.save("guide_p10", "十", {"source": "guide.md"})
    store.save("guide_p2", "旧内容", {"source": "guide.md"})
    store.save("guide_p2", "新内容", {"source": "guide.md"})
    records = store.load_content_many(["guide_p10", "guide_p2", "guide_p2"])

    assert [row["parent_id"] for row in records] == ["guide_p2", "guide_p10"]
    assert records[0]["content"] == "新内容"
    assert store.load("guide_p2.json")["page_content"] == "新内容"
    assert store.list_sources() == ["guide.md"]


def test_missing_and_corrupt_parent_are_explicit_errors(tmp_path):
    parent_dir = tmp_path / "parents"
    store = ParentStoreManager(parent_dir)

    with pytest.raises(FileNotFoundError):
        store.load("absent")
    (parent_dir / "broken.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        store.load("broken")


def test_real_qdrant_persists_and_reopens(tmp_path, monkeypatch):
    # 本日只测索引存储，直接准备标准 Document；分块已在 D05 验证。
    documents = [Document(
        page_content="response_format 错误：使用 function_calling。",
        metadata={"source": "deepseek.md", "parent_id": "deepseek_p0"},
    )]
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())

    first = VectorDbManager()
    try:
        first.create_collection(config.CHILD_COLLECTION)
        collection = first.get_collection(config.CHILD_COLLECTION)
        collection.add_documents(documents)
        before = collection.similarity_search("response_format", k=1)
        assert before[0].metadata["source"] == "deepseek.md"
    finally:
        first._VectorDbManager__client.close()

    # 同一临时路径重新打开，检验真的持久化了，而非依赖内存对象。
    reopened = VectorDbManager()
    try:
        reopened.create_collection(config.CHILD_COLLECTION)
        result = reopened.get_collection(config.CHILD_COLLECTION).similarity_search("response_format", k=1)
        assert result[0].page_content == before[0].page_content
        assert result[0].metadata == before[0].metadata
    finally:
        reopened._VectorDbManager__client.close()


def test_collection_rejects_incompatible_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    manager = VectorDbManager()
    try:
        # 先建一个 2 维 collection，再让上游按当前 5 维 embedding 校验。
        manager._VectorDbManager__client.create_collection(
            "wrong_size", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        with pytest.raises(ValueError, match="dense vector size"):
            manager.create_collection("wrong_size")
    finally:
        manager._VectorDbManager__client.close()
