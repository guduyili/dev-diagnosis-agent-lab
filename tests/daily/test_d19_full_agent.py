"""D19 / W07-A：一个文件展开完整导入、工具调用和图执行。

运行：python -m pytest tests/daily/test_d19_full_agent.py -v
建议最后读：先完成 D04/D08/D12，再逐句跟踪本文件。
知识数据、模型的三步决策和最终响应都在下面；仅模型/embedding 使用测试替身。
自己改：在最后的答案前增加一次工具调用，检查新增证据与实际调用次数。
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_qdrant import SparseEmbeddings, SparseVector

import config
import utils
import db.vector_db_manager as vector_module
import rag_agent.tools as tools_module
import rag_agent.nodes as nodes
from document_chunker import DocumentChunker
from core.document_manager import DocumentManager
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from rag_agent.schemas import QueryAnalysis

pytestmark = pytest.mark.day(19)

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


def test_full_graph_search_parent_answer_and_aggregate(tmp_path, monkeypatch):
    # 1. 来自已报告错误及本地 rewrite_query 的学习资料。
    markdown = (
        "# response_format 错误记录\n\n"
        "用户报告的实际错误文本是：This response_format type is unavailable now。\n"
        '本地 rewrite_query 使用 with_structured_output(QueryAnalysis, method="function_calling")。\n'
        "它将结构化结果放进工具参数，避免依赖 json_schema 响应格式。\n"
        "离线构造成功不能证明 DeepSeek 在线请求已经成功。\n"
    )
    path = tmp_path / "deepseek.md"
    path.write_text(markdown, encoding="utf-8")
    monkeypatch.setattr(config, "MARKDOWN_DIR", str(tmp_path / "markdown"))
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    monkeypatch.setattr(utils, "_get_token_encoding", lambda: None)
    monkeypatch.setattr(nodes, "BASE_TOKEN_THRESHOLD", 100000)  # 压缩另在 D14 检验。
    parent_store = ParentStoreManager(tmp_path / "parents")
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: parent_store)
    chunker = DocumentChunker()
    vector_db = VectorDbManager()
    try:
        # 2. 真正的 Markdown -> 分块 -> parent JSON + child 向量。
        vector_db.create_collection(config.CHILD_COLLECTION)
        manager = DocumentManager(SimpleNamespace(
            chunker=chunker, parent_store=parent_store, vector_db=vector_db,
            collection_name=config.CHILD_COLLECTION,
        ))
        assert manager.add_documents([str(path)]) == (1, 0)
        collection = vector_db.get_collection(config.CHILD_COLLECTION)
        spy = Mock(wraps=collection.similarity_search)
        monkeypatch.setattr(collection, "similarity_search", spy)
        factory = ToolFactory(collection)

        # 3. 可见的模型响应：改写一次、搜索 child、读取 parent、子答案、聚合。
        # deepseek_p0 是上述真实导入按文件名和块序号生成的 ID。
        analysis_model = Mock()
        analysis_model.invoke.side_effect = [
            QueryAnalysis(is_clear=True, questions=["response_format"], clarification_needed=""),
        ]
        decision_model = Mock()
        decision_model.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{
                "name": "search_child_chunks", "args": {"query": "response_format", "limit": 1},
                "id": "s1", "type": "tool_call",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "retrieve_parent_chunks", "args": {"parent_id": "deepseek_p0"},
                "id": "p1", "type": "tool_call",
            }]),
            AIMessage(content="子答案：function_calling\nSources:\n- deepseek.md"),
        ]
        llm = Mock()
        llm.with_structured_output.return_value = analysis_model
        llm.bind_tools.return_value = decision_model
        llm.invoke.side_effect = [AIMessage(content="最终回答：function_calling\nSources:\n- deepseek.md")]

        # 4. 直接调用真实图、真实工具，不手工代替节点更新状态。
        graph = create_agent_graph(llm, factory.create_tools())
        result = graph.invoke(
            {"messages": [HumanMessage(content="response_format 如何处理？")]},
            config={"configurable": {"thread_id": "full-flow"}},
        )

        # 5. 检查实际证据传递，而不仅是最后的固定响应字符串。
        spy.assert_called_once()
        record = result["agent_answers"][0]
        assert any("function_calling" in context for context in record["contexts"])
        assert all("File Name: deepseek.md" in context for context in record["contexts"])
        assert result["messages"][-1].content == "最终回答：function_calling\nSources:\n- deepseek.md"
        child_prompt = "\n".join(message.content for message in decision_model.invoke.call_args_list[1].args[0])
        parent_prompt = "\n".join(message.content for message in decision_model.invoke.call_args_list[2].args[0])
        assert "This response_format type is unavailable now" in child_prompt
        assert "function_calling" not in child_prompt  # 第一个 child 尚不包含修复方式。
        assert "deepseek_p0" in parent_prompt and "function_calling" in parent_prompt
        assert analysis_model.invoke.call_count == 1
        assert decision_model.invoke.call_count == 3
        assert llm.invoke.call_count == 1
    finally:
        vector_db._VectorDbManager__client.close()


def test_second_turn_clears_previous_agent_answers():
    # 此用例只检验跨轮 reset，不需要再创建知识库。
    analysis_model = Mock()
    analysis_model.invoke.side_effect = [
        QueryAnalysis(is_clear=True, questions=["response_format"], clarification_needed=""),
        QueryAnalysis(is_clear=True, questions=["parent_id"], clarification_needed=""),
    ]
    decision_model = Mock()
    decision_model.invoke.side_effect = [AIMessage(content="旧子答案"), AIMessage(content="新子答案")]
    llm = Mock()
    llm.with_structured_output.return_value = analysis_model
    llm.bind_tools.return_value = decision_model
    llm.invoke.side_effect = [AIMessage(content="旧最终答案"), AIMessage(content="新最终答案")]
    graph = create_agent_graph(llm, [])
    run_config = {"configurable": {"thread_id": "two-turns"}}

    graph.invoke({"messages": [HumanMessage(content="response_format")]}, config=run_config)
    result = graph.invoke({"messages": [HumanMessage(content="parent_id")]}, config=run_config)

    assert len(result["agent_answers"]) == 1
    assert result["agent_answers"][0]["answer"] == "新子答案"
    final_prompt = "\n".join(message.content for message in llm.invoke.call_args_list[-1].args[0])
    assert "旧子答案" not in final_prompt
    assert analysis_model.invoke.call_count == 2
    assert decision_model.invoke.call_count == 2
    assert llm.invoke.call_count == 2
