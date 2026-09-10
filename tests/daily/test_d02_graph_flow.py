"""D02 / W01-B：用三个可见的模型响应，观察真实主图的节点顺序。

运行：python -m pytest tests/daily/test_d02_graph_flow.py -v
模型只在此测试内用 unittest.mock.Mock 代替；create_agent_graph 和图状态都是真的。
自己改：让 decision_model 返回工具请求，再观察为什么还需要一个真实工具。
"""
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from rag_agent.graph import create_agent_graph
from rag_agent.schemas import QueryAnalysis

pytestmark = pytest.mark.day(2)


@pytest.mark.current_behavior
def test_real_graph_can_finish_without_tool_calls():
    # 1. 明确安排每一类模型调用返回什么。
    analysis_model = Mock()
    analysis_model.invoke.side_effect = [
        QueryAnalysis(is_clear=True, questions=["response_format"], clarification_needed=""),
    ]
    decision_model = Mock()
    decision_model.invoke.side_effect = [AIMessage(content="没有检索，无法核验。")]
    llm = Mock()
    llm.with_structured_output.return_value = analysis_model
    llm.bind_tools.return_value = decision_model
    llm.invoke.side_effect = [AIMessage(content="资料不足，待补充。")]

    # 2. 直接编译上游图，记录真实节点 update。
    graph = create_agent_graph(llm, [])
    run_config = {"configurable": {"thread_id": "day02"}, "recursion_limit": 50}
    events = list(graph.stream(
        {"messages": [HumanMessage(content="response_format")]},
        config=run_config, stream_mode="updates",
    ))

    # 3. 当前实现没有强制首步检索；这条测试记录它的现状。
    names = [name for event in events for name in event]
    assert names == ["summarize_history", "rewrite_query", "agent", "aggregate_answers"]
    snapshot = graph.get_state(run_config)
    assert snapshot.next == ()
    assert snapshot.values["messages"][-1].content == "资料不足，待补充。"
    assert snapshot.values["agent_answers"][0]["contexts"] == []
    assert analysis_model.invoke.call_count == 1
    assert decision_model.invoke.call_count == 1
    assert llm.invoke.call_count == 1
