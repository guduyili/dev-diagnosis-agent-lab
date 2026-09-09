"""让上游 project 作为被测源码导入，不把它安装成主项目的一部分。"""
from pathlib import Path
import socket
import sys

import pytest


UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "references" / "upstream" / "agentic-rag-for-dummies"
UPSTREAM_PROJECT = UPSTREAM_ROOT / "project"

if str(UPSTREAM_PROJECT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_PROJECT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "upstream_day(number): 对应上游精读路线的学习日验收（离线）",
    )


@pytest.fixture(autouse=True)
def upstream_tests_are_offline(monkeypatch):
    """防止未来新增测试意外把上游学习验收变成在线调用。"""
    def deny(*args, **kwargs):
        pytest.fail("NETWORK_BLOCKED: 上游学习测试只能使用本地替身。")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
