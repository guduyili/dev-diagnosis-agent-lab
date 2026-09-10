"""D12 / W04-C：真实检查点恢复，以及 rewrite 的真实 SDK 请求格式。

运行：python -m pytest tests/daily/test_d12_clarification.py -v
自己改：先增加一次“不充分澄清”，再给足信息；所有预设模型响应就在当前函数。
HTTP 用例使用真实客户端序列化，MockTransport 仅代替服务端，不连接 DeepSeek。
"""
import json
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

import config
from rag_agent.graph import create_agent_graph
from rag_agent.nodes import rewrite_query
from rag_agent.schemas import QueryAnalysis

pytestmark = pytest.mark.day(12)


def test_clarification_resume_uses_same_checkpoint_and_separates_explicit_thread_ids():
    # 1. A 第一次不清楚；B 的独立请求清楚；A 补充后清楚。
    analysis_model = Mock()
    analysis_model.invoke.side_effect = [
        QueryAnalysis(is_clear=False, questions=[], clarification_needed="请补充具体的产品名称或相关文件名称。"),
        QueryAnalysis(is_clear=True, questions=["parent_id"], clarification_needed=""),
        QueryAnalysis(is_clear=True, questions=["DeepSeek 如何配置"], clarification_needed=""),
    ]
    decision_model = Mock()
    decision_model.invoke.side_effect = [AIMessage(content="B 的子答案"), AIMessage(content="A 的子答案")]
    llm = Mock()
    llm.with_structured_output.return_value = analysis_model
    llm.bind_tools.return_value = decision_model
    llm.invoke.side_effect = [AIMessage(content="B 的最终答案"), AIMessage(content="A 的最终答案")]

    # 2. 直接创建真实主图。两个 thread_id 是我们显式指定的，不是 UI 自动分配。
    graph = create_agent_graph(llm, [])
    a = {"configurable": {"thread_id": "A"}}
    b = {"configurable": {"thread_id": "B"}}
    graph.invoke({"messages": [HumanMessage(content="它怎么配置？")]}, config=a)
    paused = graph.get_state(a)
    assert paused.next == ("request_clarification",)
    assert paused.values["pendingQuery"] == "它怎么配置？"

    graph.invoke({"messages": [HumanMessage(content="parent_id")]}, config=b)
    assert graph.get_state(b).values["pendingQuery"] == ""
    assert graph.get_state(a).values["pendingQuery"] == "它怎么配置？"

    # 3. 写入同一检查点，再从中断处继续，而非重新走 START。
    graph.update_state(a, {"messages": [HumanMessage(content="指 DeepSeek")]})
    graph.invoke(None, config=a)

    resumed = graph.get_state(a)
    assert resumed.next == ()
    assert resumed.values["pendingQuery"] == ""
    assert resumed.values["pendingClarifications"] == []
    b_prompt = "\n".join(message.content for message in analysis_model.invoke.call_args_list[1].args[0])
    a_prompt = "\n".join(message.content for message in analysis_model.invoke.call_args_list[2].args[0])
    assert "它怎么配置？" not in b_prompt
    assert "它怎么配置？" in a_prompt and "指 DeepSeek" in a_prompt
    assert graph.get_state(b).values["messages"][-1].content == "B 的最终答案"
    assert analysis_model.invoke.call_count == 3
    assert decision_model.invoke.call_count == 2
    assert llm.invoke.call_count == 2


def test_rewrite_uses_function_calling_and_preserves_model_error():
    analysis_model = Mock()
    analysis_model.invoke.side_effect = RuntimeError("injected provider failure")
    llm = Mock()
    llm.with_structured_output.return_value = analysis_model

    with pytest.raises(RuntimeError, match="injected provider failure"):
        rewrite_query({"messages": [HumanMessage(content="response_format")]}, llm)

    llm.with_structured_output.assert_called_once_with(QueryAnalysis, method="function_calling")
    analysis_model.invoke.assert_called_once()


def test_rewrite_serializes_real_sdk_request_at_http_boundary():
    captured = []
    # 这就是本用例服务端返回的结构化数据，明确是测试响应。
    response_data = {"is_clear": True, "questions": ["response_format"], "clarification_needed": ""}

    def transport(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert payload.get("response_format", {}).get("type") != "json_schema"
        assert payload["tools"][0]["function"]["name"] == "QueryAnalysis"
        return httpx.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": config.LLM_MODEL,
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "qa1", "type": "function", "function": {
                        "name": "QueryAnalysis", "arguments": json.dumps(response_data),
                    },
                }],
            }}],
        })

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        llm = ChatOpenAI(
            model=config.LLM_MODEL, api_key="offline-test-key",
            base_url="https://offline.invalid/v1", http_client=client, max_retries=0,
        )
        result = rewrite_query({"messages": [HumanMessage(content="response_format")]}, llm)

    assert len(captured) == 1
    assert result["questionIsClear"] is True
    assert result["rewrittenQuestions"] == ["response_format"]
