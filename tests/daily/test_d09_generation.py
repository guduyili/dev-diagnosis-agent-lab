"""D09 / W03-C：真实节点如何把问题和检索文本传给模型。

运行：python -m pytest tests/daily/test_d09_generation.py -v
输入是手写的 LangChain 消息，用于数据流实验，不冒充实际召回记录。
自己改：给工具消息换一个来源，确认捕获到的模型输入同步变化。
"""
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_agent.nodes import orchestrator, aggregate_answers, fallback_response

pytestmark = pytest.mark.day(9)


def test_orchestrator_passes_tool_evidence_back_to_model():
    evidence = ToolMessage(
        content="File Name: deepseek.md\nContent: function_calling",
        name="search_child_chunks", tool_call_id="s1",
    )
    state = {
        "question": "response_format",
        "messages": [
            HumanMessage(content="response_format", name="agent_question"),
            AIMessage(content="", tool_calls=[{
                "name": "search_child_chunks", "args": {"query": "response_format"},
                "id": "s1", "type": "tool_call",
            }]),
            evidence,
        ],
    }
    model = Mock()
    model.invoke.side_effect = [AIMessage(content="使用 function_calling。")]

    update = orchestrator(state, llm_with_tools=model)

    received_messages = model.invoke.call_args.args[0]
    assert evidence in received_messages
    assert update["iteration_count"] == 1
    assert update["tool_call_count"] == 0
    assert update["messages"][0].content == "使用 function_calling。"
    model.invoke.assert_called_once()


def test_aggregate_sorts_subanswers_and_uses_original_question():
    state = {
        "messages": [], "originalQuery": "如何修复并验证？",
        "agent_answers": [
            {"index": 1, "answer": "SECOND_ANSWER", "contexts": ["RAW_CONTEXT_NOT_PASSED"]},
            {"index": 0, "answer": "FIRST_ANSWER", "contexts": []},
        ],
    }
    model = Mock()
    model.invoke.side_effect = [AIMessage(content="合并后的输出")]

    result = aggregate_answers(state, model)

    prompt = "\n".join(message.content for message in model.invoke.call_args.args[0])
    assert "如何修复并验证？" in prompt
    assert prompt.index("FIRST_ANSWER") < prompt.index("SECOND_ANSWER")
    assert "RAW_CONTEXT_NOT_PASSED" not in prompt
    assert result["messages"][-1].content == "合并后的输出"
    model.invoke.assert_called_once()


def test_fallback_with_no_evidence_still_calls_model():
    model = Mock()
    model.invoke.side_effect = [AIMessage(content="资料不足")]

    result = fallback_response({"question": "unknown", "messages": []}, model)

    prompt = "\n".join(message.content for message in model.invoke.call_args.args[0])
    assert "No data was retrieved" in prompt
    assert result["messages"][0].content == "资料不足"
    model.invoke.assert_called_once()
