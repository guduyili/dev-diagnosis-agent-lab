"""真实上游导入、按日选择、临时存储和离线模型边界。"""
import importlib
from pathlib import Path
import re
import socket
from types import SimpleNamespace

import pytest

from .support import DATA, UPSTREAM_PROJECT, FixedDenseEmbeddings, FixedSparseEmbeddings

MANUAL_DAYS = {
    3: "真实 incident 与来源审核",
    15: "人工校准 DeepEval 裁判；上游没有内建裁判运行器",
    18: "真实任务中的使用与修正量记录",
    21: "对真实源码做红→绿回归演练",
    23: "成功、失败与回归演示",
    24: "独立讲解、个人贡献与后续计划",
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
    group = parser.getgroup("daily-learning")
    group.addoption("--day", help="只运行某学习日，例如 D06")
    group.addoption("--through-day", help="累计运行，例如 D12")


def pytest_configure(config):
    config.addinivalue_line("markers", "day(number): 学习日 D01..D24")
    config.addinivalue_line("markers", "current_behavior: 记录上游现状，不等于推荐的目标设计")
    config._daily_manual = []


def pytest_collection_modifyitems(config, items):
    selected, through = config.getoption("--day"), config.getoption("--through-day")
    if selected and through:
        raise pytest.UsageError("--day 与 --through-day 不能同时使用。")
    limit = day_number(selected or through) if selected or through else 24
    config._daily_manual = [d for d in MANUAL_DAYS if d <= limit and (not selected or d == limit)]
    if selected and limit in MANUAL_DAYS:
        raise pytest.UsageError(f"D{limit:02} 需人工证据：{MANUAL_DAYS[limit]}。见 docs/11-daily-tests.md。")
    if not selected and not through:
        return
    kept, dropped = [], []
    for item in items:
        marker = item.get_closest_marker("day")
        number = marker.args[0] if marker else None
        wanted = number is not None and (number == limit if selected else number <= limit)
        (kept if wanted else dropped).append(item)
    if not kept:
        raise pytest.UsageError("没有对应自动测试；请在命令中指定 tests/daily。")
    items[:] = kept
    config.hook.pytest_deselected(items=dropped)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.write_sep("-", "Upstream daily learning checks")
    terminalreporter.write_line(f"被测源码：{UPSTREAM_PROJECT}")
    for day in config._daily_manual:
        terminalreporter.write_line(f"MANUAL_NOT_CHECKED D{day:02}: {MANUAL_DAYS[day]}")
    terminalreporter.write_line("通过仅证明选中的离线行为；不代表在线模型质量或个人学习已完成。")


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    """在被测模块导入前阻断 Python TCP，并关闭自动追踪。"""
    def deny(*args, **kwargs):
        pytest.fail("NETWORK_BLOCKED: 每日检验只使用离线模型响应和本地存储。")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    for name in ("LANGFUSE_ENABLED", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(name, "false")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")


@pytest.fixture
def upstream(monkeypatch, offline_only):
    """正常 import 并检查文件来源，不向 sys.modules 塞入替代业务模块。"""
    assert UPSTREAM_PROJECT.is_dir(), f"缺少上游目录：{UPSTREAM_PROJECT}"
    monkeypatch.syspath_prepend(str(UPSTREAM_PROJECT))

    def load(name):
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        assert origin and Path(origin).resolve().is_relative_to(UPSTREAM_PROJECT.resolve()), (
            f"IMPORT_ORIGIN_ERROR: {name} 来自 {origin}；请在独立 pytest 进程运行 daily。"
        )
        return module
    return load


@pytest.fixture
def settings(upstream, monkeypatch, tmp_path):
    config = upstream("config")
    for key, value in {
        "MARKDOWN_DIR": str(tmp_path / "markdown"),
        "PARENT_STORE_PATH": str(tmp_path / "parents"),
        "QDRANT_DB_PATH": str(tmp_path / "qdrant"),
        "MIN_PARENT_SIZE": 80, "MAX_PARENT_SIZE": 420,
        "CHILD_CHUNK_SIZE": 160, "CHILD_CHUNK_OVERLAP": 20,
        "EXECUTION_LOGGING_ENABLED": False, "LANGFUSE_ENABLED": False,
    }.items():
        monkeypatch.setattr(config, key, value)
    return config


@pytest.fixture
def chunker(upstream, settings):
    return upstream("document_chunker").DocumentChunker()


@pytest.fixture
def utilities(upstream, settings, monkeypatch):
    module = upstream("utils")
    # 固定到已有的字符估算分支，不下载 tokenizer 数据，不替换估算函数。
    monkeypatch.setattr(module, "_get_token_encoding", lambda: None)
    return module


@pytest.fixture
def parent_store(upstream, settings, utilities, tmp_path):
    return upstream("db.parent_store_manager").ParentStoreManager(tmp_path / "parents")


@pytest.fixture
def vector_db(upstream, settings, monkeypatch):
    module = upstream("db.vector_db_manager")
    # 只替换模型构造器，保留真实 VectorDbManager / QdrantVectorStore / QdrantClient。
    monkeypatch.setattr(module, "HuggingFaceEmbeddings", lambda **kwargs: FixedDenseEmbeddings())
    monkeypatch.setattr(module, "FastEmbedSparse", lambda **kwargs: FixedSparseEmbeddings())
    manager = module.VectorDbManager()
    try:
        manager.create_collection(settings.CHILD_COLLECTION)
        yield manager
    finally:
        # 上游没有公开 close；仅通过私有 client 释放临时目录文件锁。
        manager._VectorDbManager__client.close()


@pytest.fixture
def library(upstream, settings, chunker, parent_store, vector_db):
    """真实导入流水线：MD -> parent/child -> JSON/Qdrant。"""
    rag = SimpleNamespace(chunker=chunker, parent_store=parent_store,
                          vector_db=vector_db, collection_name=settings.CHILD_COLLECTION)
    manager = upstream("core.document_manager").DocumentManager(rag)
    paths = sorted((DATA / "knowledge").glob("*.md"))
    assert paths, "没有测试资料，不能把空数据集算作通过"
    assert manager.add_documents([str(path) for path in paths]) == (len(paths), 0)
    collection = vector_db.get_collection(settings.CHILD_COLLECTION)
    return SimpleNamespace(manager=manager, rag=rag, collection=collection, paths=paths)


@pytest.fixture
def tool_factory(upstream, library, parent_store, monkeypatch):
    module = upstream("rag_agent.tools")
    # 默认参数在定义时绑定；让工具工厂使用本次测试的真实 parent store。
    monkeypatch.setattr(module, "ParentStoreManager", lambda: parent_store)
    return module.ToolFactory(library.collection)


@pytest.fixture
def nodes(upstream, utilities):
    return upstream("rag_agent.nodes")


@pytest.fixture
def graph_factory(upstream, nodes):
    return upstream("rag_agent.graph").create_agent_graph


@pytest.fixture
def analysis(upstream):
    return upstream("rag_agent.schemas").QueryAnalysis
