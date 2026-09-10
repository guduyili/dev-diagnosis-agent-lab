"""图的条件路由：把状态转换成下一个节点或并行 Send。"""

from typing import Literal
from langgraph.types import Send
from .graph_state import State, AgentState
from config import MAX_ITERATIONS, MAX_TOOL_CALLS
from core.execution_logger import log_route

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    """查询不清楚时暂停澄清，否则为每个改写问题派发一个子 Agent。"""
    # 返回类型注解只列出节点名称，但 else 分支实际返回 list[Send]。
    # Python 不强制执行该注解；理解动态派发时应以返回表达式为准。
    # 前置约定：rewrite_query 只有在 questions 非空且 is_clear 时才置 True。
    if not state.get("questionIsClear", False):
        decision = "request_clarification"
    else:
        # Send 是动态 fan-out；每个子图只拿到自己的 question 和 index，
        # 这样并行结果仍可按 index 在 aggregate_answers 中稳定排序。
        decision = [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]
    log_route("after_rewrite", decision, state)
    return decision
    
def route_after_orchestrator_call(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    """根据模型是否请求工具以及预算决定继续、降级回答或结束。"""
    # 前置约定：orchestrator 已把本轮 AIMessage 和增量计数写入 state。
    # 此函数只决定路由，不负责调用工具；真正调用位于 ToolNode。
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        # 没有工具请求就交给 collect_answer；不保证 content 非空或结论正确。
        # 空答案由 collect_answer 转为 "Unable to generate an answer."。
        decision = "collect_answer"
        log_route("after_orchestrator_call", decision, state)
        return decision

    # The counters already include the current LLM response. Allow a final
    # answer at the iteration boundary, but do not execute tool calls that
    # would exceed the configured research budget.
    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        # 计数由当前节点先写入；因此这里只允许 fallback，不执行超预算工具。
        # 默认值下的手算示例（均指合并后的计数）：
        #   iterations=9, calls=8，且有工具请求 -> tools（恰好 8 次仍允许）。
        #   iterations=9, calls=9 -> fallback（整个本轮批次都不执行）。
        #   iterations=10，且有工具请求 -> fallback；没有请求则前面已去 collect。
        # 因此 >= 与 > 的差别是有意的。fallback 自身仍会调用一次模型。
        decision = "fallback_response"
        log_route("after_orchestrator_call", decision, state)
        return decision
    
    decision = "tools"
    log_route("after_orchestrator_call", decision, state)
    return decision
