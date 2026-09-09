"""D16–D17：上游已有日志与 UI 消息转换；JUnit 解析器属于未来扩展。"""
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage


@pytest.mark.day(16)
def test_logged_node_returns_real_result_and_calls_once(upstream, settings, capsys, monkeypatch):
    logger = upstream("core.execution_logger")
    monkeypatch.setattr(settings, "EXECUTION_LOGGING_ENABLED", True)
    monkeypatch.setattr(settings, "EXECUTION_LOG_USE_COLOR", False)
    calls = []
    expected = {"retrieved_contexts": ["data"]}
    def business(state):
        calls.append(state)
        return expected
    result = logger.logged_node("learning-node", business)({"question": "response_format"})
    assert result is expected
    assert len(calls) == 1
    output = capsys.readouterr().out
    assert "NODE START: learning-node" in output
    assert "NODE OUTPUT: learning-node" in output


@pytest.mark.day(16)
def test_logged_node_reraises_same_exception(upstream, settings, capsys):
    logger = upstream("core.execution_logger")
    settings.EXECUTION_LOGGING_ENABLED = True
    failure = RuntimeError("injected failure")
    def broken(state):
        raise failure
    with pytest.raises(RuntimeError) as caught:
        logger.logged_node("broken-node", broken)({})
    assert caught.value is failure
    assert "broken-node" in capsys.readouterr().out


@pytest.mark.day(16)
def test_token_estimate_ignores_tool_call_arguments(utilities):
    from .support import call_message
    text = "abcdefgh"
    assert utilities.estimate_context_tokens([HumanMessage(content=text)]) == 2
    # 已固定到已有的 len//4 分支；不代表真实 provider 的 token 计费。
    assert utilities.estimate_context_tokens([
        call_message("search_child_chunks", {"query": "large " * 1000})
    ]) == 0


@pytest.mark.day(17)
def test_ui_tool_card_matches_call_id_and_truncates_only_display(upstream, utilities):
    module = upstream("core.chat_interface")
    ui = module.ChatInterface(None)
    messages, active = [], {}
    chunk = AIMessageChunk(content="", tool_calls=[{
        "name": "search_child_chunks", "args": {"query": "x"}, "id": "s1", "type": "tool_call",
    }])
    ui._handle_tool_call(chunk, messages, active)
    ui._handle_tool_call(chunk, messages, active)
    assert len(messages) == 1
    raw = ToolMessage(content="X" * 400, tool_call_id="s1")
    ui._handle_tool_result(raw, messages, active)
    assert "X" * 300 in messages[0]["content"]
    assert "X" * 301 not in messages[0]["content"]
    assert raw.content == "X" * 400


@pytest.mark.day(17)
def test_ui_rewrite_parser_waits_for_complete_json(upstream, utilities):
    module = upstream("core.chat_interface")
    assert module.parse_rewrite_json('{"is_clear":') is None
    parsed = module.parse_rewrite_json('{"is_clear":true,"questions":["response_format"],"clarification_needed":""}')
    assert parsed["questions"] == ["response_format"]
    assert "response_format" in module.format_rewrite_content(
        '{"is_clear":true,"questions":["response_format"]}')


@pytest.mark.day(17)
def test_optional_observability_is_disabled_without_network(upstream, settings):
    tracker = upstream("core.observability").Observability()
    assert tracker.get_handler() is None
    assert tracker.flush() is None
