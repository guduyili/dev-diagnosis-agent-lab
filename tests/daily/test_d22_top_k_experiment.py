"""D22 / W08-A：只改变 k，比较真实索引的召回结果。

运行：python -m pytest tests/daily/test_d22_top_k_experiment.py -v
自己改：增加一个含相同关键字但无关的文档；观察为何命中不能直接代表回答质量。
"""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector

import config
import db.vector_db_manager as vector_module
from db.vector_db_manager import VectorDbManager

pytestmark = pytest.mark.day(22)

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


def test_single_factor_top_k_changes_actual_retrieval_count(tmp_path, monkeypatch):
    # 标准 Document 是真实 LangChain 类型；三条资料的关键词可以手动核验。
    documents = [
        Document(page_content="response_format 错误采用 function_calling。", metadata={"source": "deepseek.md"}),
        Document(page_content="parent_id 将 child 关联到 parent。", metadata={"source": "parents.md"}),
        Document(page_content="thread_id 标识图检查点。", metadata={"source": "sessions.md"}),
    ]
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    vector_db = VectorDbManager()
    try:
        vector_db.create_collection(config.CHILD_COLLECTION)
        collection = vector_db.get_collection(config.CHILD_COLLECTION)
        collection.add_documents(documents)

        one = collection.similarity_search("response_format", k=1)
        three = collection.similarity_search("response_format", k=3)

        assert len(one) == 1 and len(three) == 3
        assert one[0].metadata["source"] == "deepseek.md"
        assert three[0].metadata["source"] == "deepseek.md"
        assert one[0].page_content == three[0].page_content
        # 此处只能验证固定测试 embedding 的行为，不证明 k=3 语义质量更高。
    finally:
        vector_db._VectorDbManager__client.close()
