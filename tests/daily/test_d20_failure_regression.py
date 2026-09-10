"""D20 / W07-B：图的预算失败与模型错误，测试中直接展开预设请求。

运行：python -m pytest tests/daily/test_d20_failure_regression.py -v
自己改：将预算从 1 改成 2，先预测真实工具会不会执行。
第一项使用真实 Qdrant collection 并记录方法调用；没有依赖共享工具 fixture。
"""
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_qdrant import SparseEmbeddings, SparseVector

import config
import db.vector_db_manager as vector_module
import rag_agent.edges as edges
import rag_agent.tools as tools_module
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from rag_agent.schemas import QueryAnalysis

pytestmark = pytest.mark.day(20)

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


def test_over_budget_batch_never_executes_search(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    monkeypatch.setattr(edges, "MAX_TOOL_CALLS", 1)
    store = ParentStoreManager(tmp_path / "parents")
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: store)
    vector_db = VectorDbManager()
    try:
        vector_db.create_collection(config.CHILD_COLLECTION)
        collection = vector_db.get_collection(config.CHILD_COLLECTION)
        spy = Mock(wraps=collection.similarity_search)
        monkeypatch.setattr(collection, "similarity_search", spy)
        factory = ToolFactory(collection)

        # 一次响应提出两个工具请求，预算为 1，应整批拒绝。
        analysis_model = Mock()
        analysis_model.invoke.side_effect = [
            QueryAnalysis(is_clear=True, questions=["response_format"], clarification_needed=""),
        ]
        decision_model = Mock()
        decision_model.invoke.side_effect = [AIMessage(content="", tool_calls=[
            {"name": "search_child_chunks", "args": {"query": "response_format"}, "id": "s1", "type": "tool_call"},
            {"name": "search_child_chunks", "args": {"query": "parent_id"}, "id": "s2", "type": "tool_call"},
        ])]
        llm = Mock()
        llm.with_structured_output.return_value = analysis_model
        llm.bind_tools.return_value = decision_model
        llm.invoke.side_effect = [AIMessage(content="预算停止，资料不足"), AIMessage(content="最终资料不足")]

        graph = create_agent_graph(llm, factory.create_tools())
        result = graph.invoke(
            {"messages": [HumanMessage(content="response_format")]},
            config={"configurable": {"thread_id": "budget"}},
        )

        spy.assert_not_called()
        assert result["agent_answers"][0]["contexts"] == []
        fallback_prompt = "\n".join(message.content for message in llm.invoke.call_args_list[0].args[0])
        assert "No data was retrieved" in fallback_prompt
        assert analysis_model.invoke.call_count == 1
        assert decision_model.invoke.call_count == 1
        assert llm.invoke.call_count == 2  # fallback 和 aggregate 各调用一次。
    finally:
        vector_db._VectorDbManager__client.close()


def test_model_error_propagates_out_of_real_graph():
    analysis_model = Mock()
    analysis_model.invoke.side_effect = RuntimeError("provider unreachable")
    llm = Mock()
    llm.with_structured_output.return_value = analysis_model
    graph = create_agent_graph(llm, [])

    with pytest.raises(RuntimeError, match="provider unreachable"):
        graph.invoke(
            {"messages": [HumanMessage(content="response_format")]},
            config={"configurable": {"thread_id": "error"}},
        )

    analysis_model.invoke.assert_called_once()
    llm.invoke.assert_not_called()  # 改写已经失败，不应继续生成最终回答。
