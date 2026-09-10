"""公共设置只有三件事：上游导入路径、学习日选择、防止意外联网。

这里不创建分块器、知识库、工具或模型；这些步骤全部写在当天 test_dXX 文件中。
"""
from pathlib import Path
import re
import socket
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[2] / "diagnosis_agent"
# print(PROJECT)
if not PROJECT.is_dir():
    raise pytest.UsageError(f"找不到上游源码：{PROJECT}")
# 让当天文件能直接写 from document_chunker import DocumentChunker。
sys.path.insert(0, str(PROJECT))

MANUAL_DAYS = {
    3: "真实 incident 与来源审核",
    15: "人工校准裁判；上游没有内建 DeepEval 运行器",
    18: "真实使用与修正量记录",
    21: "对真实源码进行红→绿回归演练",
    23: "成功、失败和回归演示",
    24: "独立讲解与个人贡献",
}


def day_number(value):
    value = value.upper()
    if re.fullmatch(r"W0?[1-8]-[ABC]", value):
        return (int(value[1:value.index("-")]) - 1) * 3 + "ABC".index(value[-1]) + 1
    if re.fullmatch(r"D?\d{1,2}", value):
        number = int(value.removeprefix("D"))
        if 1 <= number <= 24:
            return number
    raise pytest.UsageError("学习日使用 D01..D24 或 W01-A..W08-C。")


def pytest_addoption(parser):
    parser.addoption("--day", help="当天，例如 D05 或 W02-B")
    parser.addoption("--through-day", help="累计到某天，例如 D12")


def pytest_configure(config):
    config.addinivalue_line("markers", "day(number): 学习时段 D01..D24")
    config.addinivalue_line("markers", "current_behavior: 上游现状，可在明确新契约后改进")
    config._daily_manual = []


def pytest_collection_modifyitems(config, items):
    selected, through = config.getoption("--day"), config.getoption("--through-day")
    if selected and through:
        raise pytest.UsageError("--day 与 --through-day 不能同时使用。")
    limit = day_number(selected or through) if selected or through else 24
    config._daily_manual = [d for d in MANUAL_DAYS if d <= limit and (not selected or d == limit)]
    if selected and limit in MANUAL_DAYS:
        raise pytest.UsageError(f"D{limit:02} 需人工证据：{MANUAL_DAYS[limit]}，见 docs/11-daily-tests.md。")
    if not selected and not through:
        # 直接指定单个 test_dXX 文件时，不显示其他学习日的人工提醒。
        days = {item.get_closest_marker("day").args[0] for item in items if item.get_closest_marker("day")}
        if days != set(range(1, 25)) - set(MANUAL_DAYS):
            config._daily_manual = []
        return
    kept, dropped = [], []
    for item in items:
        marker = item.get_closest_marker("day")
        number = marker.args[0] if marker else None
        wanted = number is not None and (number == limit if selected else number <= limit)
        (kept if wanted else dropped).append(item)
    if not kept:
        raise pytest.UsageError("没有对应测试，请指定 tests/daily。")
    items[:] = kept
    config.hook.pytest_deselected(items=dropped)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.write_sep("-", "每日模块实验")
    for day in config._daily_manual:
        terminalreporter.write_line(f"MANUAL_NOT_CHECKED D{day:02}: {MANUAL_DAYS[day]}")
    terminalreporter.write_line("自动通过不等于在线效果通过，也不等于已独立完成学习。")


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    """唯一公共 fixture：禁止常用 Python TCP 连接；不是业务对象工厂。"""
    def deny(*args, **kwargs):
        pytest.fail("NETWORK_BLOCKED：本组实验只在明确标注的模型边界使用离线响应。")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    for name in ("LANGFUSE_ENABLED", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(name, "false")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
