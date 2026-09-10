"""D01 / W01-A：核对真实配置、结构化输出协议，以及系统组装。

运行：python -m pytest tests/daily/test_d01_config_and_system.py -v
前两项适合先学；最后一项展开系统初始化，模型/embedding 替换点也写在本文件。
自己改：先仅调整一个配置；不要把“初始化通过”当成“在线模型可用”。
"""
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector

import config
import db.vector_db_manager as vector_module
import core.rag_system as system_module
import rag_agent.tools as tools_module
from document_chunker import DocumentChunker
from db.parent_store_manager import ParentStoreManager
from core.rag_system import RAGSystem
from rag_agent.schemas import QueryAnalysis

pytestmark = pytest.mark.day(1)

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


def test_config_is_from_upstream_and_sizes_are_valid():
    expected = Path(__file__).resolve().parents[2] / "references/upstream/agentic-rag-for-dummies/project/config.py"
    assert Path(config.__file__).resolve() == expected.resolve()
    assert 0 <= config.CHILD_CHUNK_OVERLAP < config.CHILD_CHUNK_SIZE
    assert 0 < config.MIN_PARENT_SIZE <= config.MAX_PARENT_SIZE
    assert config.MAX_TOOL_CALLS > 0 and config.MAX_ITERATIONS > 0
    assert DocumentChunker() is not None  # 真正运行构造器的校验。


def test_query_schema_requires_clarification_field():
    parsed = QueryAnalysis(is_clear=True, questions=["response_format"], clarification_needed="")
    assert parsed.questions == ["response_format"]
    with pytest.raises(ValidationError):
        QueryAnalysis(is_clear=True, questions=["response_format"])


def test_rag_system_wires_config_and_resets_checkpoint(tmp_path, monkeypatch):
    # 1. 路径和模型替换都明确写在此处，实际库仍为 Qdrant。
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    monkeypatch.setattr(config, "LANGFUSE_ENABLED", False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(config, "LLM_API_KEY", "offline-test-key")
    parent_store = ParentStoreManager(tmp_path / "parents")
    monkeypatch.setattr(system_module, "ParentStoreManager", lambda: parent_store)
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: parent_store)
    llm = Mock()
    chat_constructor = Mock(return_value=llm)
    monkeypatch.setattr(system_module, "ChatOpenAI", chat_constructor)

    # 2. 创建并初始化上游 RAGSystem，不替换 create_agent_graph。
    system = RAGSystem()
    try:
        system.initialize()

        arguments = chat_constructor.call_args.kwargs
        assert arguments["model"] == config.LLM_MODEL
        assert arguments["base_url"] == config.LLM_BASE_URL
        bound_tools = llm.bind_tools.call_args.args[0]
        assert {tool.name for tool in bound_tools} == {"search_child_chunks", "retrieve_parent_chunks"}
        run_config = system.get_config()
        assert "callbacks" not in run_config
        assert run_config["recursion_limit"] == config.GRAPH_RECURSION_LIMIT

        # 3. 真实检查点写入、删除、切换 thread。
        previous_id = system.thread_id
        system.agent_graph.update_state(run_config, {"pendingQuery": "old pending"})
        system.reset_thread()

        assert system.thread_id != previous_id
        assert system.agent_graph.get_state(run_config).values == {}
        assert system.agent_graph.get_state(system.get_config()).values == {}
    finally:
        # 上游没有公开 close，用实际 client 释放临时数据库的 Windows 文件锁。
        system.vector_db._VectorDbManager__client.close()
