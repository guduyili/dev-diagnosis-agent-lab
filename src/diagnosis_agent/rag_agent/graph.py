"""构建两层 LangGraph：主会话图和单问题 Agent 子图。"""

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial

from .graph_state import State, AgentState
from core.execution_logger import logged_node
from .nodes import (
    aggregate_answers,
    collect_answer,
    compress_context,
    fallback_response,
    orchestrator,
    request_clarification,
    rewrite_query,
    should_compress_context,
    summarize_history,
)
from .edges import route_after_orchestrator_call, route_after_rewrite

def create_agent_graph(llm, tools_list):
    """绑定工具、注册节点和边，并返回带内存检查点的主图。

    主图负责会话摘要、查询改写和最终聚合；子图负责一个 rewritten question
    的检索循环。主图用 ``Send`` 并行派发多个问题，子图结束后统一汇总。
    """
    # bind_tools 只描述模型可以请求什么；实际执行由 LangGraph ToolNode 完成。
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    # 检查点保存到进程内存，并由运行配置中的 thread_id 标识会话。
    # 它使“暂停 -> update_state -> 继续”成为可能，进程重启后不持久保留。
    checkpointer = InMemorySaver()

    print("Compiling agent graph...")
    # --- 子图：一个问题的“模型 → 工具 → 压缩/模型”循环 ---
    agent_builder = StateGraph(AgentState)
    # partial 固定 llm 参数后，图运行时只需传 state。
    # logged_node 包装的是“输入 state -> 输出 update”，日志不会代替 reducer。
    agent_builder.add_node("orchestrator", logged_node("agent.orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools)))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("compress_context", logged_node("agent.compress_context", partial(compress_context, llm=llm)))
    agent_builder.add_node("fallback_response", logged_node("agent.fallback_response", partial(fallback_response, llm=llm)))
    agent_builder.add_node("should_compress_context", logged_node("agent.should_compress_context", should_compress_context))
    agent_builder.add_node("collect_answer", logged_node("agent.collect_answer", collect_answer))

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
    agent_builder.add_edge("tools", "should_compress_context")
    # should_compress_context 自己返回 Command(update=..., goto=...)，同时更新
    # 状态和选择下一步。因此这里没有再为它注册一条无条件出边。
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)

    agent_subgraph = agent_builder.compile()

    # 主图正常路径：
    # START -> summarize_history -> rewrite_query -> Send(agent, ...)
    #       -> aggregate_answers -> END
    # 澄清路径：rewrite_query -> [暂停在 request_clarification 之前]
    # 恢复后：request_clarification -> rewrite_query；不会重新走历史摘要节点。
    # --- 主图：一次会话的历史处理、澄清和答案聚合 ---
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", logged_node("main.summarize_history", partial(summarize_history, llm=llm)))
    graph_builder.add_node("rewrite_query", logged_node("main.rewrite_query", partial(rewrite_query, llm=llm)))
    graph_builder.add_node("request_clarification", logged_node("main.request_clarification", request_clarification))
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", logged_node("main.aggregate_answers", partial(aggregate_answers, llm=llm)))

    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    # 这里进入聚合时依赖 agent_answers 已合并各子任务结果。子图完成次序可能
    # 不同，collect_answer 将 question_index 存为结果的 index，聚合据此排序。
    graph_builder.add_edge("aggregate_answers", END)

    # request_clarification 本身不生成状态；interrupt_before 让 UI 有机会等待
    # 用户补充信息，然后用同一 thread_id 更新 messages 并恢复执行。
    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

    # compile 只建立可运行的图；工具协议、外部 API 和实际召回尚未在此验证。
    print("✓ Agent graph compiled successfully.")
    return agent_graph
