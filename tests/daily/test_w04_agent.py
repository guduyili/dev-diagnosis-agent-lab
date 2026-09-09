"""D10–D12：真实状态合并、路由、ToolNode 和检查点恢复。"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from pydantic import ValidationError

from .support import ScriptedLLM, call_message, prompt_text


@pytest.mark.day(10)
def test_reducers_are_applied_by_real_langgraph(upstream):
    module = upstream("rag_agent.graph_state")
    builder = StateGraph(module.AgentState)
    builder.add_node("step", lambda state: {
        "tool_call_count": 1, "iteration_count": 1, "retrieved_contexts": ["a", "b"],
        "retrieval_keys": {"search::q", "parent::p0"},
    })
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    result = builder.compile().invoke({
        "messages": [], "tool_call_count": 2, "iteration_count": 3,
        "retrieved_contexts": ["a"], "retrieval_keys": {"search::q"},
    })
    assert result["tool_call_count"] == 3
    assert result["iteration_count"] == 4
    assert result["retrieved_contexts"] == ["a", "b"]
    assert result["retrieval_keys"] == {"search::q", "parent::p0"}


@pytest.mark.day(10)
def test_answer_reducer_empty_update_is_not_reset(upstream):
    reducer = upstream("rag_agent.graph_state").accumulate_or_reset
    old = [{"index": 0, "answer": "上一轮"}]
    assert reducer(old, []) == old
    assert reducer(old, [{"__reset__": True}]) == []
    assert reducer(old, [{"index": 1, "answer": "本轮"}])[-1]["answer"] == "本轮"


@pytest.mark.day(10)
def test_real_send_contains_question_and_index(upstream, utilities):
    route = upstream("rag_agent.edges").route_after_rewrite
    assert route({"questionIsClear": False}) == "request_clarification"
    sends = route({"questionIsClear": True, "rewrittenQuestions": ["response_format", "parent_id"]})
    assert all(isinstance(item, Send) and item.node == "agent" for item in sends)
    assert [item.arg["question_index"] for item in sends] == [0, 1]
    assert [item.arg["question"] for item in sends] == ["response_format", "parent_id"]


@pytest.mark.day(11)
@pytest.mark.parametrize("iterations,calls,has_tool,expected", [
    (9, 8, True, "tools"), (9, 9, True, "fallback_response"),
    (10, 1, True, "fallback_response"), (10, 9, False, "collect_answer"),
])
def test_budget_boundary_uses_proposed_calls(upstream, utilities, monkeypatch, iterations, calls, has_tool, expected):
    edges = upstream("rag_agent.edges")
    # 固定实验预算，不依赖学习者之后修改默认配置的数值。
    monkeypatch.setattr(edges, "MAX_ITERATIONS", 10)
    monkeypatch.setattr(edges, "MAX_TOOL_CALLS", 8)
    message = call_message("search_child_chunks", {"query": "response_format"}) if has_tool else AIMessage(content="done")
    result = edges.route_after_orchestrator_call({
        "messages": [message], "iteration_count": iterations, "tool_call_count": calls,
    })
    assert result == expected


@pytest.mark.day(11)
def test_toolnode_matches_real_result_to_call_id(tool_factory, upstream):
    node = ToolNode(tool_factory.create_tools())
    # 当前版本 ToolNode 从图运行时取得 context，放入真实 StateGraph 再调用。
    builder = StateGraph(upstream("rag_agent.graph_state").AgentState)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    state = builder.compile().invoke({"messages": [
        call_message("retrieve_parent_chunks", {"parent_id": "parents_p0"}, "parent-call"),
    ]})
    result = state["messages"][-1]
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "parent-call"
    assert result.name == "retrieve_parent_chunks"
    assert "page_content" in result.content


@pytest.mark.day(11)
def test_tool_schema_rejects_missing_query(tool_factory):
    search = tool_factory.create_tools()[0]
    with pytest.raises(ValidationError):
        search.invoke({"limit": 1})


@pytest.mark.day(11)
def test_missing_parent_and_search_failure_return_error_sentinels(tool_factory, monkeypatch):
    assert tool_factory._retrieve_parent_chunks("missing").startswith("PARENT_RETRIEVAL_ERROR:")
    def fail_search(*args, **kwargs):
        raise RuntimeError("injected index unavailable")
    monkeypatch.setattr(tool_factory.collection, "similarity_search", fail_search)
    assert tool_factory._search_child_chunks("response_format") == "RETRIEVAL_ERROR: injected index unavailable"


@pytest.mark.day(12)
def test_clarification_resume_uses_same_checkpoint_and_separates_explicit_thread_ids(graph_factory, analysis):
    # 真正 compile/invoke/update_state；不模拟 checkpoint 或状态合并。
    model = ScriptedLLM(
        analyses=[
            analysis(is_clear=False, questions=[], clarification_needed="请补充具体的产品名称或相关文件名称。"),
            analysis(is_clear=True, questions=["parent_id"], clarification_needed=""),
            analysis(is_clear=True, questions=["DeepSeek 如何配置"], clarification_needed=""),
        ],
        decisions=[AIMessage(content="B 的子答案"), AIMessage(content="A 的子答案")],
        answers=[AIMessage(content="B 的最终答案"), AIMessage(content="A 的最终答案")],
    )
    graph = graph_factory(model, [])
    a = {"configurable": {"thread_id": "A"}}
    b = {"configurable": {"thread_id": "B"}}
    graph.invoke({"messages": [HumanMessage(content="它怎么配置？")]}, config=a)
    paused = graph.get_state(a)
    assert paused.next == ("request_clarification",)
    assert paused.values["pendingQuery"] == "它怎么配置？"
    graph.invoke({"messages": [HumanMessage(content="parent_id")]}, config=b)
    assert graph.get_state(b).values["pendingQuery"] == ""
    assert graph.get_state(a).values["pendingQuery"] == "它怎么配置？"

    graph.update_state(a, {"messages": [HumanMessage(content="指 DeepSeek")]})
    graph.invoke(None, config=a)
    resumed = graph.get_state(a)
    assert resumed.next == ()
    assert resumed.values["pendingQuery"] == ""
    assert resumed.values["pendingClarifications"] == []
    analyses = [call for call in model.calls if call["kind"] == "analysis"]
    assert "它怎么配置？" not in prompt_text(analyses[1])
    assert "它怎么配置？" in prompt_text(analyses[2]) and "指 DeepSeek" in prompt_text(analyses[2])
    assert graph.get_state(b).values["messages"][-1].content == "B 的最终答案"
    model.assert_consumed()


@pytest.mark.day(12)
def test_rewrite_uses_function_calling_and_preserves_model_error(nodes, analysis):
    model = ScriptedLLM(analyses=[RuntimeError("injected provider failure")])
    with pytest.raises(RuntimeError, match="injected provider failure"):
        nodes.rewrite_query({"messages": [HumanMessage(content="response_format")]}, model)
    assert model.structured_options == [(analysis, {"method": "function_calling"})]
    # 这里只检验节点调用约定，不把它说成 HTTP 请求体或在线兼容性已经验证。


@pytest.mark.day(12)
def test_rewrite_serializes_real_sdk_request_at_http_boundary(nodes, settings):
    import json
    import httpx
    from langchain_openai import ChatOpenAI
    captured = []

    def transport(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert payload.get("response_format", {}).get("type") != "json_schema"
        assert payload["tools"][0]["function"]["name"] == "QueryAnalysis"
        # 仅模拟服务端响应；HTTP 请求体由真实 ChatOpenAI/OpenAI SDK 序列化。
        return httpx.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": settings.LLM_MODEL,
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "qa1", "type": "function", "function": {
                        "name": "QueryAnalysis",
                        "arguments": json.dumps({"is_clear": True, "questions": ["response_format"],
                                                 "clarification_needed": ""}),
                    },
                }],
            }}],
        })

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        llm = ChatOpenAI(model=settings.LLM_MODEL, api_key="offline-test-key",
                         base_url="https://offline.invalid/v1", http_client=client, max_retries=0)
        result = nodes.rewrite_query({"messages": [HumanMessage(content="response_format")]}, llm)
    assert len(captured) == 1
    assert result["questionIsClear"] is True
    assert result["rewrittenQuestions"] == ["response_format"]
