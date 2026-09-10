"""D10 / W04-A：手算状态更新，再交给真实 reducer 和 LangGraph 验证。

运行：python -m pytest tests/daily/test_d10_state_and_routes.py -v
自己改：把本轮增量从 1 改成 2，先算出结果；再尝试重复上下文和空更新。
"""
import pytest
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from rag_agent.graph_state import AgentState, accumulate_or_reset
from rag_agent.edges import route_after_rewrite

pytestmark = pytest.mark.day(10)


def test_reducers_are_applied_by_real_langgraph():
    previous = {
        "messages": [], "tool_call_count": 2, "iteration_count": 3,
        "retrieved_contexts": ["a"], "retrieval_keys": {"search::q"},
    }
    increment = {
        "tool_call_count": 1, "iteration_count": 1,
        "retrieved_contexts": ["a", "b"], "retrieval_keys": {"search::q", "parent::p0"},
    }
    # 测试节点只提供增量，合并规则来自上游 AgentState。
    builder = StateGraph(AgentState)
    builder.add_node("step", lambda state: increment)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)

    result = builder.compile().invoke(previous)

    assert result["tool_call_count"] == 3
    assert result["iteration_count"] == 4
    assert result["retrieved_contexts"] == ["a", "b"]
    assert result["retrieval_keys"] == {"search::q", "parent::p0"}


def test_answer_reducer_empty_update_is_not_reset():
    old = [{"index": 0, "answer": "上一轮"}]
    assert accumulate_or_reset(old, []) == old
    assert accumulate_or_reset(old, [{"__reset__": True}]) == []
    assert accumulate_or_reset(old, [{"index": 1, "answer": "本轮"}])[-1]["answer"] == "本轮"


def test_real_send_contains_question_and_index():
    assert route_after_rewrite({"questionIsClear": False}) == "request_clarification"

    sends = route_after_rewrite({
        "questionIsClear": True, "rewrittenQuestions": ["response_format", "parent_id"],
    })

    assert all(isinstance(item, Send) and item.node == "agent" for item in sends)
    assert [item.arg["question_index"] for item in sends] == [0, 1]
    assert [item.arg["question"] for item in sends] == ["response_format", "parent_id"]
