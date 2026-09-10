"""D16 / W06-A：日志包装器和 token 估算；上游没有 JUnit 解析器。

运行：python -m pytest tests/daily/test_d16_logging.py -v
自己改：让被包装的函数返回不同对象或抛不同异常，检查输出和异常身份。
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import config
import utils
from core.execution_logger import logged_node
from utils import estimate_context_tokens

pytestmark = pytest.mark.day(16)


def test_logged_node_returns_real_result_and_calls_once(capsys, monkeypatch):
    monkeypatch.setattr(config, "EXECUTION_LOGGING_ENABLED", True)
    monkeypatch.setattr(config, "EXECUTION_LOG_USE_COLOR", False)
    calls = []
    expected = {"retrieved_contexts": ["data"]}

    def business(state):
        calls.append(state)
        return expected

    result = logged_node("learning-node", business)({"question": "response_format"})

    assert result is expected
    assert len(calls) == 1
    output = capsys.readouterr().out
    assert "NODE START: learning-node" in output
    assert "NODE OUTPUT: learning-node" in output


def test_logged_node_reraises_same_exception(capsys, monkeypatch):
    monkeypatch.setattr(config, "EXECUTION_LOGGING_ENABLED", True)
    failure = RuntimeError("injected failure")

    def broken(state):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        logged_node("broken-node", broken)({})

    assert caught.value is failure
    assert "broken-node" in capsys.readouterr().out


def test_token_estimate_ignores_tool_call_arguments(monkeypatch):
    # 明确测试现有 len//4 回退分支；没有把它当成真实 provider token 用量。
    monkeypatch.setattr(utils, "_get_token_encoding", lambda: None)
    content_message = HumanMessage(content="abcdefgh")
    request_message = AIMessage(content="", tool_calls=[{
        "name": "search_child_chunks", "args": {"query": "large " * 1000},
        "id": "s1", "type": "tool_call",
    }])

    assert estimate_context_tokens([content_message]) == 2
    assert estimate_context_tokens([request_message]) == 0
