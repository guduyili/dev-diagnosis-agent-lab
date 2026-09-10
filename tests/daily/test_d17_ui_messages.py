"""D17 / W06-B：直接测试 UI 消息转换；不启动 Gradio 页面。

运行：python -m pytest tests/daily/test_d17_ui_messages.py -v
自己改：增加两个相同工具名但不同 ID 的调用，确认输出没有填到错误卡片。
"""
import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

import config
from core.chat_interface import ChatInterface, parse_rewrite_json, format_rewrite_content
from core.observability import Observability

pytestmark = pytest.mark.day(17)


def test_ui_tool_card_matches_call_id_and_truncates_only_display():
    ui = ChatInterface(None)  # 本用例只调用转换函数，不执行 chat 或模型请求。
    messages, active = [], {}
    request = AIMessageChunk(content="", tool_calls=[{
        "name": "search_child_chunks", "args": {"query": "x"},
        "id": "s1", "type": "tool_call",
    }])
    result = ToolMessage(content="X" * 400, tool_call_id="s1")

    ui._handle_tool_call(request, messages, active)
    ui._handle_tool_call(request, messages, active)
    ui._handle_tool_result(result, messages, active)

    assert len(messages) == 1
    assert "X" * 300 in messages[0]["content"]
    assert "X" * 301 not in messages[0]["content"]
    assert result.content == "X" * 400


def test_ui_rewrite_parser_waits_for_complete_json():
    incomplete = '{"is_clear":'
    complete = '{"is_clear":true,"questions":["response_format"],"clarification_needed":""}'

    assert parse_rewrite_json(incomplete) is None
    assert parse_rewrite_json(complete)["questions"] == ["response_format"]
    assert "response_format" in format_rewrite_content(complete)


def test_optional_observability_is_disabled_without_network(monkeypatch):
    monkeypatch.setattr(config, "LANGFUSE_ENABLED", False)

    tracker = Observability()

    assert tracker.get_handler() is None
    assert tracker.flush() is None
