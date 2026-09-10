"""轻量终端执行日志。

日志的目标是回答“图走了哪些节点、工具实际收到了什么、状态如何变化”，
部分文本字段会截断、消息只展示最近几条，以便阅读。截断不是脱敏，
工具参数、异常等字段仍可能完整打印；日志本身也不是答案正确性的证明。
"""

from __future__ import annotations

from datetime import datetime
from pprint import pformat
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

import config


COLORS = {
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "magenta": "\033[95m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _enabled() -> bool:
    """读取运行时开关；使用 getattr 兼容旧配置。"""
    # 缺少配置属性时回退为 True；当前 config 的显式值优先，并非恒定开启。
    return bool(getattr(config, "EXECUTION_LOGGING_ENABLED", True))


def _color(text: str, color: str) -> str:
    if not getattr(config, "EXECUTION_LOG_USE_COLOR", True):
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _truncate(value: Any, max_chars: int | None = None) -> str:
    """截断长文本并标注省略字符数，保持日志可读。"""
    # 单位是字符，不是 token；max_chars=0 为假值，会回退到默认长度而非输出空串。
    text = "" if value is None else str(value)
    limit = max_chars or getattr(config, "EXECUTION_LOG_MAX_CHARS", 1200)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def _message_role(message: Any) -> str:
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, RemoveMessage):
        return "remove"
    return message.__class__.__name__


def _message_preview(message: Any) -> dict[str, Any]:
    preview = {
        "type": _message_role(message),
        "id": getattr(message, "id", None),
    }

    if isinstance(message, RemoveMessage):
        return preview

    content = getattr(message, "content", "")
    if content:
        preview["content"] = _truncate(content)

    tool_calls = getattr(message, "tool_calls", None)
    # content 的截断不影响这里的 args；参数按原值进入日志摘要。
    if tool_calls:
        preview["tool_calls"] = [
            {
                "name": call.get("name"),
                "args": call.get("args"),
                "id": call.get("id"),
            }
            for call in tool_calls
        ]

    tool_name = getattr(message, "name", None)
    if tool_name:
        preview["name"] = tool_name

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        preview["tool_call_id"] = tool_call_id

    return preview


def _messages_preview(messages: list[Any]) -> dict[str, Any]:
    # count 是总数，last_messages 只有最后四条消息；四条消息不一定是两轮对话，
    # 也可能全是工具交互。日志未展示的消息不等于已从图状态删除。
    return {
        "count": len(messages),
        "last_messages": [_message_preview(message) for message in messages[-4:]],
    }


def state_preview(state: Any) -> dict[str, Any]:
    """将 LangGraph state 转成不易爆炸的可读摘要。"""
    # 特判已知大字段，其他键直接保留；本函数既不验证 schema，也不修改原 state。
    if not isinstance(state, dict):
        return {"value": _truncate(state)}

    preview: dict[str, Any] = {}

    for key, value in state.items():
        if key == "messages":
            preview[key] = _messages_preview(value or [])
        elif key in {"conversation_summary", "context_summary", "final_answer"}:
            preview[key] = _truncate(value)
        elif key == "agent_answers":
            preview[key] = {
                "count": len(value or []),
                "items": [
                    {
                        "index": item.get("index"),
                        "question": _truncate(item.get("question", ""), 240),
                        "answer": _truncate(item.get("answer", ""), 500),
                    }
                    for item in (value or [])[:3]
                    if isinstance(item, dict)
                ],
            }
        elif isinstance(value, set):
            preview[key] = sorted(value)
        else:
            preview[key] = value

    return preview


def update_preview(update: Any) -> Any:
    """将节点局部 update 转成日志摘要。"""
    # Command 不是 dict，会退化为字符串预览。日志中看不清 goto/update 的结构
    # 不代表真实返回值丢失；logged_node 最后返回的仍是原 result。
    if not isinstance(update, dict):
        return _truncate(update)

    preview: dict[str, Any] = {}
    for key, value in update.items():
        if key == "messages":
            preview[key] = [_message_preview(message) for message in value]
        elif key in {"conversation_summary", "context_summary", "final_answer"}:
            preview[key] = _truncate(value)
        elif key == "agent_answers":
            preview[key] = state_preview({"agent_answers": value})["agent_answers"]
        elif isinstance(value, set):
            preview[key] = sorted(value)
        else:
            preview[key] = value
    return preview


def _print_block(title: str, payload: Any, color: str) -> None:
    """统一打印带时间、标题和颜色的日志块。"""
    if not _enabled():
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(_color(f"\n[{timestamp}] {title}", color))
    print(_color("-" * 80, "dim"))
    print(pformat(payload, width=120, sort_dicts=False))


def log_chat_start(message: str, thread_id: str, has_pending_interrupt: bool) -> None:
    _print_block(
        "USER QUERY",
        {
            "thread_id": thread_id,
            "pending_interrupt": has_pending_interrupt,
            "message": _truncate(message),
        },
        "blue",
    )


def log_chat_end(state: Any) -> None:
    _print_block("FINAL GRAPH STATE", state_preview(state), "blue")


def log_node_start(name: str, state: Any) -> None:
    _print_block(f"NODE START: {name}", state_preview(state), "cyan")


def log_node_end(name: str, update: Any) -> None:
    _print_block(f"NODE OUTPUT: {name}", update_preview(update), "green")


def log_route(name: str, decision: Any, state: Any | None = None) -> None:
    payload = {"decision": _truncate(decision)}
    if state is not None:
        payload["state"] = state_preview(state)
    _print_block(f"ROUTE: {name}", payload, "yellow")


def log_tool_start(name: str, args: dict[str, Any]) -> None:
    _print_block(f"TOOL START: {name}", args, "magenta")


def log_tool_end(name: str, output: Any) -> None:
    _print_block(f"TOOL OUTPUT: {name}", {"output": _truncate(output)}, "magenta")


def log_error(scope: str, error: Exception) -> None:
    _print_block(f"ERROR: {scope}", {"type": error.__class__.__name__, "message": str(error)}, "red")


def logged_node(name: str, fn):
    """返回一个包装器，在节点前后记录状态，异常时记录后继续抛出。"""
    # 同步包装器：调用 fn 一次、不重试、不 await。NODE OUTPUT 是节点返回的
    # 局部更新，尚未合入 reducer；理解最终状态还需结合 graph_state.py。
    # 裸 raise 保留原异常 traceback，避免把执行失败伪装成一个成功的空返回值。
    def _wrapped(state, *args, **kwargs):
        log_node_start(name, state)
        try:
            result = fn(state, *args, **kwargs)
        except Exception as exc:
            log_error(name, exc)
            raise
        log_node_end(name, result)
        return result

    return _wrapped
