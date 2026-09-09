"""D13–D14：从真实工具和节点提取评测资料；上游没有 DeepEval adapter。"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph

from .support import ScriptedLLM, call_message, prompt_text


@pytest.mark.day(13)
def test_context_extraction_splits_actual_search_output_and_filters_errors(nodes, tool_factory, settings):
    output = tool_factory._search_child_chunks("parent_id", limit=2)
    messages = [
        ToolMessage(content=output, name="search_child_chunks", tool_call_id="s1"),
        ToolMessage(content=output, name="search_child_chunks", tool_call_id="s2"),
        ToolMessage(content="NO_RELEVANT_CHUNKS", tool_call_id="empty"),
        ToolMessage(content="RETRIEVAL_ERROR: broken", tool_call_id="err"),
        AIMessage(content="模型猜测，不是工具证据"),
    ]
    contexts = nodes._retrieval_contexts(messages)
    assert contexts == output.split(settings.CHILD_CHUNK_SEPARATOR)
    assert len(contexts) == 2
    assert all("File Name:" in value for value in contexts)


@pytest.mark.day(13)
def test_collect_answer_keeps_actual_tool_contexts(nodes, tool_factory):
    output = tool_factory._retrieve_parent_chunks("deepseek_p0")
    contexts = nodes._retrieval_contexts([ToolMessage(content=output, tool_call_id="p1")])
    result = nodes.collect_answer({
        "messages": [AIMessage(content="实际传给 collect_answer 的输出")],
        "question": "response_format", "question_index": 2, "retrieved_contexts": contexts,
    })
    record = result["agent_answers"][0]
    assert record["index"] == 2
    assert record["answer"] == "实际传给 collect_answer 的输出"
    assert record["contexts"] == [output]
    assert record["question"] == "response_format"


@pytest.mark.day(13)
@pytest.mark.parametrize("message", [AIMessage(content=""), call_message("search_child_chunks", {"query": "x"})])
def test_collect_does_not_treat_empty_answer_or_tool_request_as_answer(nodes, message):
    result = nodes.collect_answer({"messages": [message], "question": "x", "question_index": 0})
    assert result["final_answer"] == "Unable to generate an answer."


@pytest.mark.day(14)
def test_compression_removes_messages_but_retains_evidence_via_real_reducer(nodes, upstream, monkeypatch):
    # 测真实节点 + 真 reducer；仅模型总结结果预先安排。
    module = upstream("rag_agent.graph_state")
    evidence = "File Name: deepseek.md\nContent: method=function_calling"
    model = ScriptedLLM(answers=[AIMessage(content="压缩后的检索事实")])
    builder = StateGraph(module.AgentState)
    builder.add_node("compress", lambda state: nodes.compress_context(state, model))
    builder.add_edge(START, "compress")
    builder.add_edge("compress", END)
    result = builder.compile().invoke({
        "question": "response_format",
        "messages": [
            HumanMessage(content="response_format", name="agent_question", id="q"),
            call_message("search_child_chunks", {"query": "response_format"}, "s1"),
            ToolMessage(content=evidence, name="search_child_chunks", tool_call_id="s1", id="t"),
        ],
        "retrieved_contexts": [evidence], "retrieval_keys": {"search::response_format"},
    })
    assert [message.id for message in result["messages"]] == ["q"]
    assert result["retrieved_contexts"] == [evidence]
    assert "压缩后的检索事实" in result["context_summary"]
    assert "response_format" in result["context_summary"]
    assert evidence in prompt_text(model.calls[0])
    model.assert_consumed()


@pytest.mark.day(14)
@pytest.mark.current_behavior
def test_failed_request_is_still_recorded_as_retrieval_key(nodes, monkeypatch):
    monkeypatch.setattr(nodes, "BASE_TOKEN_THRESHOLD", 10000)
    command = nodes.should_compress_context({
        "messages": [
            call_message("retrieve_parent_chunks", {"parent_id": "missing"}, "p1"),
            ToolMessage(content="PARENT_RETRIEVAL_ERROR: missing", tool_call_id="p1"),
        ],
    })
    assert command.goto == "orchestrator"
    assert command.update["retrieval_keys"] == {"parent::missing"}
    assert command.update["retrieved_contexts"] == []


@pytest.mark.day(14)
@pytest.mark.parametrize("threshold,expected", [(1, "compress_context"), (10000, "orchestrator")])
def test_compression_threshold_controls_actual_command(nodes, monkeypatch, threshold, expected):
    monkeypatch.setattr(nodes, "BASE_TOKEN_THRESHOLD", threshold)
    command = nodes.should_compress_context({
        "messages": [HumanMessage(content="response_format " * 10)],
    })
    assert command.goto == expected


@pytest.mark.day(14)
def test_history_summary_receives_old_messages_and_preserves_recent_ones(nodes, upstream, monkeypatch):
    monkeypatch.setattr(nodes, "PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP", 3)
    model = ScriptedLLM(answers=[AIMessage(content="旧对话摘要")])
    builder = StateGraph(upstream("rag_agent.graph_state").State)
    builder.add_node("summarize", lambda state: nodes.summarize_history(state, model))
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", END)
    messages = [HumanMessage(content="OLD_QUESTION", id="1"), AIMessage(content="OLD_REPLY", id="2"),
                HumanMessage(content="RECENT_QUESTION", id="3"), AIMessage(content="RECENT_REPLY", id="4"),
                HumanMessage(content="CURRENT_QUESTION", id="5")]
    result = builder.compile().invoke({"messages": messages, "agent_answers": [{"answer": "旧子答案"}]})
    assert [message.id for message in result["messages"]] == ["3", "4", "5"]
    assert result["conversation_summary"] == "旧对话摘要"
    assert result["agent_answers"] == []
    assert "OLD_QUESTION" in prompt_text(model.calls[0])
    assert "CURRENT_QUESTION" not in prompt_text(model.calls[0])
    model.assert_consumed()
