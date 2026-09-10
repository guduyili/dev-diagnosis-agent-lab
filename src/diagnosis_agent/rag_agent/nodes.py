"""主图和 Agent 子图的节点实现。

这里的函数都遵循 LangGraph 的约定：读取 state，返回一个局部 update，
而不是直接修改整个 state。主图节点处理会话和问题拆分；子图节点处理一次
检索研究。阅读本文件时，建议同时对照 graph.py 的边和 graph_state.py 的
reducer，才能看懂一个返回值最终如何合并进图状态。
"""

from typing import Literal, Set
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command
from .graph_state import State, AgentState
from .schemas import QueryAnalysis
from .prompts import *
from utils import estimate_context_tokens
from config import BASE_TOKEN_THRESHOLD, CHILD_CHUNK_SEPARATOR, MAIN_HISTORY_MESSAGES_TO_KEEP, TOKEN_GROWTH_FACTOR

if MAIN_HISTORY_MESSAGES_TO_KEEP < 2:
    raise ValueError("MAIN_HISTORY_MESSAGES_TO_KEEP must be at least 2.")

PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP = max(MAIN_HISTORY_MESSAGES_TO_KEEP - 1, 0)
# 默认保留 4 条普通消息，回答生成前先留 3 条，为新 AI 回复预留 1 个位置。
# 这是消息条数而非对话轮数；配置校验保证 keep_count>=1，避免 [:-0] 等切片陷阱。

def _is_plain_conversation_message(msg) -> bool:
    """判断消息是否应进入用户可见的普通对话历史。

    工具消息、带 tool_calls 的 AI 消息和内部命名消息只服务于图内部，不能
    被当成用户上一轮自然语言，否则摘要和后续改写会重复暴露执行细节。
    """
    return (
        isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)
        and not getattr(msg, "name", None)
    )

def _name_internal_message(message, name):
    """Tag a subgraph-only message so it is not treated as chat history."""
    # model_copy 保留原消息 id，仅修改 name。交给消息 reducer 后，相同 id 的
    # 消息可被替换为带内部标签的版本；无需再额外追加一份澄清回复。
    # name 是本项目的约定，不是“模型私有思考”的证明或权限隔离机制。
    return message.model_copy(update={"name": name})

def _retrieval_contexts(messages) -> list[str]:
    """从工具消息提取实际检索上下文，并过滤错误/空结果哨兵。"""
    # search 返回多个 child 的拼接串，按共享分隔符拆开；parent 返回整段文本。
    # 去重使用完整文本相等，不会合并不同版本的近似段落。
    # 过滤只看字符串前缀：不是通用 ToolResult 校验，也不验证来源支持答案。
    contexts = []
    ignored_prefixes = (
        "NO_RELEVANT_CHUNKS",
        "NO_PARENT_DOCUMENT",
        "RETRIEVAL_ERROR:",
        "PARENT_RETRIEVAL_ERROR:",
    )
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content).strip()
        if content and not content.startswith(ignored_prefixes):
            parts = content.split(CHILD_CHUNK_SEPARATOR) if message.name == "search_child_chunks" else [content]
            contexts.extend(part for part in parts if part)
    return list(dict.fromkeys(contexts))

def _format_conversation(messages) -> str:
    """将消息转成摘要模型易读的 User/Assistant 文本。"""
    # 调用者已筛出 HumanMessage/AIMessage；函数自身把所有非 Human 标为 Assistant。
    lines = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)

def _remove_messages_not_in(messages, keep_ids):
    """用 RemoveMessage 标记不再需要的消息，而不是直接修改原列表。"""
    # 不删除 SystemMessage 或缺 id 的消息；所以这不是无条件的“只留 keep_ids”。
    # 图的消息通道通常会分配 id；直接用手写无 id 消息测试时要考虑此差异。
    removals = []
    for msg in messages:
        msg_id = getattr(msg, "id", None)
        if isinstance(msg, SystemMessage) or not msg_id:
            continue
        if msg_id not in keep_ids:
            removals.append(RemoveMessage(id=msg_id))
    return removals

def _recent_conversation(messages, pending_query="") -> list:
    """Return recent context before the current user message.

    During clarification, exclude the unresolved query and the assistant's
    clarification request because they are represented explicitly.
    """
    # 调用约定：当前用户输入是最后一条普通消息，[:-1] 排除它以免重复注入。
    # 有 pending_query 时，回溯其最后一次出现位置，只留它之前的上下文。
    # 匹配使用 strip 后的完整字符串相等，同一句问题重复出现时以最后一次为准。
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]
    recent_messages = plain_messages[:-1]

    if pending_query:
        for index in range(len(recent_messages) - 1, -1, -1):
            msg = recent_messages[index]
            if isinstance(msg, HumanMessage) and str(msg.content).strip() == pending_query:
                return recent_messages[:index]

    return recent_messages

def summarize_history(state: State, llm):
    """压缩旧对话并为新一轮回答清空上一轮子 Agent 结果。

    只保留最近的少量普通消息；更早内容交给摘要模型。即使本轮无需摘要，
    ``agent_answers`` 的 reset 标记也要返回，否则上一轮答案会被聚合到本轮。
    """
    messages = state.get("messages", [])
    updates = {"agent_answers": [{"__reset__": True}]}

    if not messages:
        return updates

    # 先过滤内部消息，再计算“旧消息”和“保留消息”，避免工具输出占用历史预算。
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]
    keep_count = PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP
    messages_to_summarize = plain_messages[:-keep_count] if len(plain_messages) > keep_count else []
    keep_ids = {getattr(msg, "id", None) for msg in plain_messages[-keep_count:]}
    keep_ids.discard(None)

    # RemoveMessage 和 conversation_summary 一起作为 update 返回后才合并入图；
    # 本地变量 messages 仍保留旧对象，因此本次摘要可以读到即将删除的内容。
    removals = _remove_messages_not_in(messages, keep_ids)
    if removals:
        updates["messages"] = removals

    if not messages_to_summarize:
        # 没有旧内容时省掉一次 LLM 调用，但仍返回 reset 和可能的删除操作。
        return updates

    existing_summary = state.get("conversation_summary", "").strip()
    conversation = "Existing summary:\n"
    conversation += f"{existing_summary or '(none)'}\n\n"
    conversation += "New messages to merge into the summary:\n"
    conversation += _format_conversation(messages_to_summarize)

    summary_response = llm.invoke([
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=conversation),
    ])
    updates["conversation_summary"] = summary_response.content.strip()
    # 摘要是模型生成的有损文本；这里没有事实核验或严格字数裁剪。
    return updates

def rewrite_query(state: State, llm):
    """将当前问题改写成一个或多个自包含检索问题，或提出澄清请求。

    pendingQuery 存在时，当前消息被视为用户对澄清问题的补充，而不是一个全新
    问题；这也是澄清后能够恢复原始意图的关键。结构化输出显式使用
    ``function_calling``，将 schema 放进工具定义，避免依赖 json_schema 响应格式。
    """
    # 入口必须有 messages，且最后一条是新用户问题或澄清补充。
    # state["..."] 是必需字段读取；state.get(..., 默认值) 用于尚未出现的历史字段。
    last_message = state["messages"][-1]
    current_query = str(last_message.content).strip()
    conversation_summary = state.get("conversation_summary", "").strip()
    pending_query = state.get("pendingQuery", "").strip()
    pending_clarifications = state.get("pendingClarifications", [])
    recent_messages = _recent_conversation(state["messages"], pending_query)

    # 只向模型注入必要的摘要、近期上下文和当前问题，避免把整个消息历史再次
    # 发送给模型，也避免让工具输出影响查询改写。
    context_parts = []
    if conversation_summary:
        context_parts.append(f"Conversation Summary:\n{conversation_summary}")
    if recent_messages:
        context_parts.append(f"Recent Conversation:\n{_format_conversation(recent_messages)}")

    if pending_query:
        # 示例：pendingQuery="它怎么配置？"，用户补充="指 DeepSeek"。
        # 原问题与补充显式放在同一个 prompt 中，而不是只检索“指 DeepSeek”。
        clarifications = [*pending_clarifications, current_query]
        clarification_text = "\n".join(
            f"{index}. {value}" for index, value in enumerate(clarifications, start=1)
        )
        context_parts.append(
            f"Unresolved User Query:\n{pending_query}\n\n"
            f"User Clarifications:\n{clarification_text}"
        )
        original_query = f"{pending_query}\nClarifications:\n{clarification_text}"
    else:
        clarifications = []
        context_parts.append(f"User Query:\n{current_query}")
        original_query = current_query

    context_section = "\n\n".join(context_parts)
    # 本 fork 针对此前 response_format 400 采用 function_calling：让模型通过
    # 工具参数表达结构化结果。它仍依赖服务端工具调用支持，不保证任意模型都兼容。
    llm_with_structure = llm.with_structured_output(
        QueryAnalysis,
        method="function_calling",
    )
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])
    # 这是本节点的一次真实模型请求。HTTP 400、超时或 schema 解析失败会抛出；
    # 没有在节点内重试或把异常转成“需要澄清”，最终由 chat() 的异常分支展示。
    clarification_message_update = (
        [_name_internal_message(last_message, "clarification_response")]
        if pending_query else []
    )

    if response.questions and response.is_clear:
        # 两个条件同时满足才派发子任务；但仍未检查每个字符串是否仅含空白。
        # 清空 pending 字段表示这轮澄清已经解决，防止下一轮继续沿用旧问题。
        return {
            "questionIsClear": True,
            "originalQuery": original_query,
            "pendingQuery": "",
            "pendingClarifications": [],
            "rewrittenQuestions": response.questions,
            "messages": clarification_message_update,
        }

    # 未达到“清楚且有问题列表”时使用澄清分支；长度阈值按字符计算，不是 token。
    # <=10 个字符的提示会被默认英语句子替换，即使短中文提示本身已经有意义。
    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    return {
        "questionIsClear": False,
        "originalQuery": "",
        "pendingQuery": pending_query or current_query,
        "pendingClarifications": clarifications,
        "rewrittenQuestions": [],
        "messages": clarification_message_update + [
            AIMessage(content=clarification, name="clarification")
        ],
    }

def request_clarification(state: State):
    """澄清节点本身不写文本；图在此节点之前中断，等待用户补充。"""
    # 提问的 AIMessage 已在 rewrite_query 中加入。恢复时此节点返回 {}，
    # 沿固定边重新进入 rewrite_query；{} 不会清空已有 state。
    return {}

# --- Agent Nodes ---
def orchestrator(state: AgentState, llm_with_tools):
    """驱动一次子问题的工具循环。

    第一次调用用提示词要求先搜索 child chunks，后续调用由模型决定继续搜索、
    回取 parent 或直接回答。返回的计数是本次模型响应提出的调用数/迭代增量，
    由 AgentState 的 reducer 累加。
    """
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        # 这里只添加提示文本，没有 tool_choice="required" 或直接执行搜索。
        # 模型仍可能不调用工具；后面的 route/collect 也没有硬性首检索校验。
        human_msg = HumanMessage(content=state["question"], name="agent_question")
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        response = _name_internal_message(response, "agent_response")
        # force_search 本身不写回 state；只在此次 invoke 中出现。
        # 返回 tool_call_count=len(response.tool_calls) 是“提出”的数量，
        # 此刻尚未经过路由预算判断和 ToolNode 执行。
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    # 后续轮次把工具消息和之前的 AI tool_calls 一起传回模型，让它能判断
    # 已经检索过什么、是否需要更大 parent 上下文或可以结束。
    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    response = _name_internal_message(response, "agent_response")
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def fallback_response(state: AgentState, llm):
    """预算耗尽后，把已有摘要和工具数据交给模型生成有限回答。"""
    # “有限”指停止继续调用检索工具，不是停止所有模型调用。本节点还会调用 llm。
    # 这里的无工具 llm 与绑定工具的 llm_with_tools 不同，不期望再产生工具请求。
    seen = set()
    unique_contents = []
    # 同一工具结果可能因循环或消息恢复出现多次；去重可减少 fallback prompt。
    for m in state["messages"]:
        # 当前实现会保留错误/空结果哨兵，没有调用 _retrieval_contexts 的前缀过滤。
        # 因此收集到 ToolMessage 不等于收集到有效证据。
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    # 只传 SystemMessage + 新的 HumanMessage，没有传入被预算拒绝的 AI tool call。
    # 这样不会留下“模型请求了工具但没有对应 ToolMessage”的协议配对问题。
    response = _name_internal_message(response, "agent_response")
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    """更新已检索 key，并根据估算 token 数选择压缩或继续循环。"""
    messages = state["messages"]

    # 本节点接在 ToolNode 后，但 key 仍来自最近 AI 请求的参数，不检查工具结果。
    # 例如 parent 文件不存在，parent::该ID 也会被记录；不能据此判断成功。
    # 这份记录主要供提示词避免重复，不是工具执行层的去重锁。
    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    # 与旧集合求并集是幂等操作：虽然 reducer 还会求一次并集，也不会重复增加。
    updated_ids = state.get("retrieval_keys", set()) | new_ids

    # token 估算只用于预算路由，不是 provider 精确计费；不同模型 tokenizer
    # 可能不同，因此这里只作相对长度保护。
    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)
    # 默认例子：摘要估算 1000 token -> 阈值 2000+900=2900。
    # 总量包含 messages+summary；动态阈值略随已有摘要增长，不是模型上下文上限。

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(
        # 先把尚未压缩的有效工具文本写入独立通道，再跳转；压缩后仍能用于复盘。
        update={
            "retrieval_keys": updated_ids,
            "retrieved_contexts": _retrieval_contexts(messages),
        },
        goto=goto,
    )

def compress_context(state: AgentState, llm):
    """把工具循环压缩成摘要，并移除已处理的旧消息。

    摘要之外额外附上已请求的 parent/query 列表，提示模型避免重复研究。
    删除消息不会删除 retrieved_contexts，它由 reducer 保留。
    """
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    # 第一条通常是用户问题；它单独写入 prompt，循环只压缩后续 AI/Tool 消息。
    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        # 下方固定提示使用 "Already executed" 字样，但 key 的收集没有验证成功。
        # 阅读时以真实 ToolMessage 为准；此处只是保留现有提示词内容，未修复协议。
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    # 留下第一条 agent_question，删除后续 AI/Tool 消息，同时保存摘要。
    # 下一次 orchestrator 因 messages 非空进入续轮分支，并注入 context_summary。
    # 删除一段完整的工具对话，也避免只保留调用请求或结果之一破坏消息配对。
    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    """把子图最后一条 AI 文本转换为统一的 agent_answers 记录。"""
    # 这里只验证消息类型、内容是否为真值及无 tool_calls，不验证正确性/忠实度。
    # 全空白但非空的 content 仍可能通过；质量验证应由单独评测承担。
    last_message = state["messages"][-1]
    # 如果边界上仍是 tool call，不能把工具请求当成答案；用明确 fallback 文本
    # 让上层知道本轮没有得到可用答案。
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": answer,
            "contexts": state.get("retrieved_contexts", []),
            # 保存的是曾经返回的有效检索文本，不是最终聚合模型的实际输入全文。
        }]
    }
# --- End of Agent Nodes---

def aggregate_answers(state: State, llm):
    """按问题 index 排序多个子 Agent 答案，再生成用户最终可见回复。"""
    # 先清理内部消息，只留下最近普通历史；最终 AIMessage 不带 name，
    # 下一轮会被 _is_plain_conversation_message 作为普通对话保留。
    messages = state.get("messages", [])
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]
    keep_ids = {getattr(msg, "id", None) for msg in plain_messages[-PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP:]}
    keep_ids.discard(None)
    removals = _remove_messages_not_in(messages, keep_ids)

    if not state.get("agent_answers"):
        return {"messages": removals + [AIMessage(content="No answers were generated.")]}

    # 并行 Send 的完成顺序不稳定，必须按原始问题 index 排序后再交给聚合模型。
    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        # 聚合 prompt 只拼接子答案 answer，没有传入各项 contexts 的原始证据。
        # 因而“聚合阶段忠实度”和“子 Agent 检索召回”是不同的评测对象。
        formatted_answers += (f"\nRetrieved response {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    # 即使只有一个子问题，也再调用一次聚合模型。最终输出仍可能有模型误差，
    # 不能因为上游子答案带 Sources，就认为聚合结果已做过来源核验。
    return {"messages": removals + [AIMessage(content=synthesis_response.content)]}
