"""D19–D22：真实图 + 真实工具/本地库的回归，模型只在边界提供预设响应。"""
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from .support import ScriptedLLM, call_message, prompt_text


@pytest.mark.day(19)
def test_full_graph_search_parent_answer_and_aggregate(graph_factory, tool_factory, analysis, monkeypatch, nodes):
    # 本例聚焦完整检索链，压缩另由 D14 验证；固定高阈值避免两种实验混在一起。
    monkeypatch.setattr(nodes, "BASE_TOKEN_THRESHOLD", 100000)
    spy = Mock(wraps=tool_factory.collection.similarity_search)
    monkeypatch.setattr(tool_factory.collection, "similarity_search", spy)
    model = ScriptedLLM(
        analyses=[analysis(is_clear=True, questions=["response_format"], clarification_needed="")],
        decisions=[
            call_message("search_child_chunks", {"query": "response_format", "limit": 1}, "s1"),
            call_message("retrieve_parent_chunks", {"parent_id": "deepseek_p0"}, "p1"),
            AIMessage(content="子答案：function_calling\nSources:\n- deepseek.md"),
        ],
        answers=[AIMessage(content="最终回答：function_calling\nSources:\n- deepseek.md")],
    )
    graph = graph_factory(model, tool_factory.create_tools())
    result = graph.invoke({"messages": [HumanMessage(content="response_format 如何处理？")]},
                          config={"configurable": {"thread_id": "full-flow"}})
    spy.assert_called_once()
    record = result["agent_answers"][0]
    assert any("function_calling" in context for context in record["contexts"])
    assert all("File Name: deepseek.md" in context for context in record["contexts"])
    assert result["messages"][-1].content == "最终回答：function_calling\nSources:\n- deepseek.md"
    decisions = [call for call in model.calls if call["kind"] == "decision"]
    assert "This response_format type is unavailable now" in prompt_text(decisions[1])
    # 首个 child 包含错误现象；parent 补回修复方式，检验父子检索的实际增益。
    assert "function_calling" not in prompt_text(decisions[1])
    assert "deepseek_p0" in prompt_text(decisions[2])
    assert "function_calling" in prompt_text(decisions[2])
    model.assert_consumed()


@pytest.mark.day(19)
def test_second_turn_clears_previous_agent_answers(graph_factory, analysis):
    model = ScriptedLLM(
        analyses=[analysis(is_clear=True, questions=[question], clarification_needed="")
                  for question in ["response_format", "parent_id"]],
        decisions=[AIMessage(content="旧子答案"), AIMessage(content="新子答案")],
        answers=[AIMessage(content="旧最终答案"), AIMessage(content="新最终答案")],
    )
    graph = graph_factory(model, [])
    cfg = {"configurable": {"thread_id": "two-turns"}}
    graph.invoke({"messages": [HumanMessage(content="response_format")]}, config=cfg)
    result = graph.invoke({"messages": [HumanMessage(content="parent_id")]}, config=cfg)
    assert len(result["agent_answers"]) == 1
    assert result["agent_answers"][0]["answer"] == "新子答案"
    assert "旧子答案" not in prompt_text(model.calls[-1])
    model.assert_consumed()


@pytest.mark.day(20)
def test_over_budget_batch_never_executes_search(graph_factory, tool_factory, upstream, analysis, monkeypatch):
    edges = upstream("rag_agent.edges")
    monkeypatch.setattr(edges, "MAX_TOOL_CALLS", 1)
    spy = Mock(wraps=tool_factory.collection.similarity_search)
    monkeypatch.setattr(tool_factory.collection, "similarity_search", spy)
    batch = AIMessage(content="", tool_calls=[
        {"name": "search_child_chunks", "args": {"query": query}, "id": f"s{i}", "type": "tool_call"}
        for i, query in enumerate(["response_format", "parent_id"])
    ])
    model = ScriptedLLM(
        analyses=[analysis(is_clear=True, questions=["response_format"], clarification_needed="")],
        decisions=[batch],
        answers=[AIMessage(content="预算停止，资料不足"), AIMessage(content="最终资料不足")],
    )
    result = graph_factory(model, tool_factory.create_tools()).invoke(
        {"messages": [HumanMessage(content="response_format")]},
        config={"configurable": {"thread_id": "budget"}},
    )
    spy.assert_not_called()
    assert result["agent_answers"][0]["contexts"] == []
    assert "No data was retrieved" in prompt_text(model.calls[2])
    model.assert_consumed()


@pytest.mark.day(20)
def test_model_error_propagates_out_of_real_graph(graph_factory):
    model = ScriptedLLM(analyses=[RuntimeError("provider unreachable")])
    graph = graph_factory(model, [])
    with pytest.raises(RuntimeError, match="provider unreachable"):
        graph.invoke({"messages": [HumanMessage(content="response_format")]},
                     config={"configurable": {"thread_id": "error"}})


@pytest.mark.day(22)
def test_single_factor_top_k_changes_actual_retrieval_count(library):
    one = library.collection.similarity_search("response_format", k=1)
    three = library.collection.similarity_search("response_format", k=3)
    assert len(one) == 1 and len(three) == 3
    assert one[0].metadata["source"] == "deepseek.md"
    assert three[0].metadata["source"] == "deepseek.md"
    assert one[0].page_content == three[0].page_content
    # 只验证固定词表 embedding 下的检索行为，不据此宣称 k=3 质量优于 k=1。
