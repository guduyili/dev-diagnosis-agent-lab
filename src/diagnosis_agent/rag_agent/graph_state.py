"""LangGraph 主图和子 Agent 使用的状态模型。

LangGraph 更新 state 时，会依据 ``Annotated`` 中的 reducer 合并字段。理解
这里的“追加/并集/累加”非常重要：它们不是普通 Python 赋值，也是工具计数和
并行子问题结果能够汇总的原因。
"""

from typing import List, Annotated, Set
from langgraph.graph import MessagesState
import operator

# 阅读入口：graph.py 注册这些状态；nodes.py 返回局部更新；这里决定如何合并。
# MessagesState 是 TypedDict 风格的状态声明，其 messages 通道使用消息合并规则：
#   新 ID -> 追加；相同 ID -> 替换该消息；RemoveMessage(id=...) -> 删除对应消息。
# 因此返回 {"messages": []} 通常不会清空历史，删除要显式返回 RemoveMessage。
# 这里不是 Pydantic 模型：类型提示不能代替运行时校验，也不能把下面的 = []
# 看成每次 Send 都自动补齐全部业务字段。读取可缺失字段时，节点使用 state.get。

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """追加回答，收到 reset 标记时清空上一轮的回答。"""
    # 示例：existing=[A]，new=[B] -> [A,B]；new=[] -> [A]（不会清空）。
    # 任一新元素带真值 __reset__ 时整批返回 []，即使同批还带答案也会被丢弃。
    # summarize_history 每轮先发 reset；并行子图再各自追加一个答案。
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    """对检索动作 key 求并集；只做去重，不验证执行成功或强制禁止重试。"""
    # {"search::x"} | {"search::x","parent::p0"} 只保留两个不同字符串。
    # key 来自模型请求参数；应与真实 ToolMessage 的成功/错误状态分开理解。
    return a | b

def append_unique(existing: List[str], new: List[str]) -> List[str]:
    """按首次出现顺序追加文本，并去除完全相同的上下文。"""
    # 利用 dict 的插入顺序：["a","b"] + ["b","c"] -> ["a","b","c"]。
    # 比较的是完整字符串；同一事实换种表述、同文不同元数据仍可能保留多份。
    return list(dict.fromkeys(existing + new))

class State(MessagesState):
    """主图状态：会话摘要、澄清信息和多个子问题的答案。"""
    # rewrite_query 写入，route_after_rewrite 读取；清楚不代表知识库一定有答案。
    questionIsClear: bool = False
    # 跨用户轮次的旧对话摘要；与子图的 context_summary（检索摘要）不同。
    conversation_summary: str = ""
    # 供最终聚合使用的用户问题；澄清成功后包含原问题和补充信息。
    originalQuery: str = ""
    # 澄清期间保留原问题和每次补充；问题清楚后两个字段都重置为空。
    pendingQuery: str = ""
    pendingClarifications: List[str] = []
    # rewrite 输出的独立问题，edges.py 按列表位置创建 Send。
    rewrittenQuestions: List[str] = []
    # 多个子图共同写同一通道，必须用 reducer 合并，而不能靠最后写入覆盖。
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

class AgentState(MessagesState):
    """单个子问题状态：检索上下文、工具循环和最终答案。"""
    # 每个 Send 给出自己的 question/index/messages；index 决定最终展示顺序，
    # 不是执行次序或完成次序。
    question: str = ""
    question_index: int = 0
    # compress_context 生成的工作摘要，下一轮作为附加消息再次送给模型。
    context_summary: str = ""
    # 前缀区分动作："search::查询文本" 和 "parent::父块ID"。
    retrieval_keys: Annotated[Set[str], set_union] = set()
    # 持续保存工具返回的有效文本，避免压缩 messages 后丢失评测原始资料。
    # 它不等同于最后一次 LLM 调用收到的完整 prompt。
    retrieved_contexts: Annotated[List[str], append_unique] = []
    final_answer: str = ""
    # collect_answer 生成一项列表；回到主图时由主图的同名 reducer 汇总。
    agent_answers: List[dict] = []
    # 节点返回的是本次增量，operator.add 会与已有值相加；不要返回累计总数。
    # 已有 2，本轮提出 3 个工具调用，返回 3 -> 新值 5；此时工具尚未执行。
    # 预算是每个子图自己的，不是整次请求所有并行子图的总预算。
    tool_call_count: Annotated[int, operator.add] = 0
    # 只统计 orchestrator 的决策轮次；rewrite/压缩/fallback/聚合调用不计在此。
    iteration_count: Annotated[int, operator.add] = 0
