"""D14 / W05-B：把状态、工具消息、模型摘要展开，观察真实压缩和历史裁剪。

运行：python -m pytest tests/daily/test_d14_compression.py -v
自己改：改变阈值或保留条数；先预测消息 ID、原始证据和摘要分别会如何变化。
"""
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph

import utils
import rag_agent.nodes as nodes
from rag_agent.graph_state import State, AgentState
from rag_agent.nodes import compress_context, should_compress_context, summarize_history

pytestmark = pytest.mark.day(14)


def test_compression_removes_messages_but_retains_evidence_via_real_reducer():
    # 1. 手写的工具消息是本次控制流输入，不是线上检索成果。
    evidence = "File Name: deepseek.md\nContent: method=function_calling"
    state = {
        "question": "response_format",
        "messages": [
            HumanMessage(content="response_format", name="agent_question", id="q"),
            AIMessage(content="", tool_calls=[{
                "name": "search_child_chunks", "args": {"query": "response_format"},
                "id": "s1", "type": "tool_call",
            }]),
            ToolMessage(content=evidence, name="search_child_chunks", tool_call_id="s1", id="t"),
        ],
        "retrieved_contexts": [evidence], "retrieval_keys": {"search::response_format"},
    }
    model = Mock()
    model.invoke.side_effect = [AIMessage(content="压缩后的检索事实")]

    # 2. 实际调用上游节点，并让真实 AgentState reducer 处理删除标记。
    builder = StateGraph(AgentState)
    builder.add_node("compress", lambda current: compress_context(current, model))
    builder.add_edge(START, "compress")
    builder.add_edge("compress", END)
    result = builder.compile().invoke(state)

    assert [message.id for message in result["messages"]] == ["q"]
    assert result["retrieved_contexts"] == [evidence]
    assert "压缩后的检索事实" in result["context_summary"]
    assert "response_format" in result["context_summary"]
    prompt = "\n".join(message.content for message in model.invoke.call_args.args[0])
    assert evidence in prompt
    model.invoke.assert_called_once()


@pytest.mark.current_behavior
def test_failed_request_is_still_recorded_as_retrieval_key(monkeypatch):
    # 仅固定到上游已有字符估算分支，不下载 tokenizer 数据。
    monkeypatch.setattr(utils, "_get_token_encoding", lambda: None)
    monkeypatch.setattr(nodes, "BASE_TOKEN_THRESHOLD", 10000)
    state = {"messages": [
        AIMessage(content="", tool_calls=[{
            "name": "retrieve_parent_chunks", "args": {"parent_id": "missing"},
            "id": "p1", "type": "tool_call",
        }]),
        ToolMessage(content="PARENT_RETRIEVAL_ERROR: missing", tool_call_id="p1"),
    ]}

    command = should_compress_context(state)

    assert command.goto == "orchestrator"
    assert command.update["retrieval_keys"] == {"parent::missing"}
    assert command.update["retrieved_contexts"] == []


@pytest.mark.parametrize("threshold,expected", [(1, "compress_context"), (10000, "orchestrator")])
def test_compression_threshold_controls_actual_command(monkeypatch, threshold, expected):
    monkeypatch.setattr(utils, "_get_token_encoding", lambda: None)
    monkeypatch.setattr(nodes, "BASE_TOKEN_THRESHOLD", threshold)
    state = {"messages": [HumanMessage(content="response_format " * 10)]}

    command = should_compress_context(state)

    assert command.goto == expected


def test_history_summary_receives_old_messages_and_preserves_recent_ones(monkeypatch):
    monkeypatch.setattr(nodes, "PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP", 3)
    model = Mock()
    model.invoke.side_effect = [AIMessage(content="旧对话摘要")]
    messages = [
        HumanMessage(content="OLD_QUESTION", id="1"),
        AIMessage(content="OLD_REPLY", id="2"),
        HumanMessage(content="RECENT_QUESTION", id="3"),
        AIMessage(content="RECENT_REPLY", id="4"),
        HumanMessage(content="CURRENT_QUESTION", id="5"),
    ]

    builder = StateGraph(State)
    builder.add_node("summarize", lambda current: summarize_history(current, model))
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", END)
    result = builder.compile().invoke({"messages": messages, "agent_answers": [{"answer": "旧子答案"}]})

    assert [message.id for message in result["messages"]] == ["3", "4", "5"]
    assert result["conversation_summary"] == "旧对话摘要"
    assert result["agent_answers"] == []
    prompt = "\n".join(message.content for message in model.invoke.call_args.args[0])
    assert "OLD_QUESTION" in prompt
    assert "CURRENT_QUESTION" not in prompt
    model.invoke.assert_called_once()
