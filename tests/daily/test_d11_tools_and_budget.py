"""D11 / W04-B：上游工具协议、真实 ToolNode 和预算路由。

运行：python -m pytest tests/daily/test_d11_tools_and_budget.py -v
自己改：增加未知工具名、坏 parent JSON 或“本轮提出多个调用”的情况。
本日 parent 工具直接访问临时 JSON；检索错误用明确的异常注入检验 catch 分支。
"""
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

import rag_agent.edges as edges
import rag_agent.tools as tools_module
from rag_agent.tools import ToolFactory
from rag_agent.graph_state import AgentState
from db.parent_store_manager import ParentStoreManager

pytestmark = pytest.mark.day(11)


@pytest.mark.parametrize("iterations,calls,has_tool,expected", [
    (9, 8, True, "tools"),
    (9, 9, True, "fallback_response"),
    (10, 1, True, "fallback_response"),
    (10, 9, False, "collect_answer"),
])
def test_budget_boundary_uses_proposed_calls(monkeypatch, iterations, calls, has_tool, expected):
    monkeypatch.setattr(edges, "MAX_ITERATIONS", 10)
    monkeypatch.setattr(edges, "MAX_TOOL_CALLS", 8)
    calls_data = [{
        "name": "search_child_chunks", "args": {"query": "response_format"},
        "id": "s1", "type": "tool_call",
    }] if has_tool else []
    message = AIMessage(content="" if has_tool else "done", tool_calls=calls_data)

    result = edges.route_after_orchestrator_call({
        "messages": [message], "iteration_count": iterations, "tool_call_count": calls,
    })

    assert result == expected  # 注意：无工具请求的分支先于预算判断。


def test_toolnode_matches_real_result_to_call_id(tmp_path, monkeypatch):
    # 1. 数据由真实 ParentStoreManager 写入；本用例不做 child 搜索。
    store = ParentStoreManager(tmp_path / "parents")
    store.save("parents_p0", "JSON 保存 page_content 和 metadata。", {"source": "parents.md"})
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: store)
    factory = ToolFactory(collection=None)
    request = AIMessage(content="", tool_calls=[{
        "name": "retrieve_parent_chunks", "args": {"parent_id": "parents_p0"},
        "id": "parent-call", "type": "tool_call",
    }])

    # 2. ToolNode 需要运行时，把它放入真实图。没有替代 StateGraph 或 ToolMessage。
    builder = StateGraph(AgentState)
    builder.add_node("tools", ToolNode(factory.create_tools()))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    result = builder.compile().invoke({"messages": [request]})["messages"][-1]

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "parent-call"
    assert result.name == "retrieve_parent_chunks"
    assert "page_content" in result.content
    assert "File Name: parents.md" in result.content


def test_tool_schema_rejects_missing_query(tmp_path, monkeypatch):
    store = ParentStoreManager(tmp_path / "parents")
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: store)
    factory = ToolFactory(collection=None)  # 参数校验失败，应在查询 collection 前停止。
    search = factory.create_tools()[0]

    with pytest.raises(ValidationError):
        search.invoke({"limit": 1})


def test_missing_parent_and_search_failure_return_error_sentinels(tmp_path, monkeypatch):
    store = ParentStoreManager(tmp_path / "parents")
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: store)
    # 只注入底层检索异常，异常转字符串的逻辑仍执行上游 ToolFactory。
    collection = Mock(spec=["similarity_search"])
    collection.similarity_search.side_effect = RuntimeError("injected index unavailable")
    factory = ToolFactory(collection)

    missing = factory._retrieve_parent_chunks("missing")
    failed_search = factory._search_child_chunks("response_format")

    assert missing.startswith("PARENT_RETRIEVAL_ERROR:")
    assert failed_search == "RETRIEVAL_ERROR: injected index unavailable"
    collection.similarity_search.assert_called_once()
