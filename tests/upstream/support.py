"""上游模块测试的轻量替身。

这些替身只负责隔离网络、Qdrant 和 LangGraph 运行时；断言仍然调用上游的
真实函数，不把主项目实现或固定答案注入被测模块。
"""
from __future__ import annotations

import sys
import types


def install_langgraph_stubs() -> None:
    """在未安装 LangGraph 的主项目环境中提供状态声明和 Send 数据对象。"""
    try:
        import langgraph.graph  # noqa: F401
    except ModuleNotFoundError:
        graph = types.ModuleType("langgraph.graph")

        class MessagesState(dict):
            pass

        graph.MessagesState = MessagesState
        langgraph = types.ModuleType("langgraph")
        langgraph.__path__ = []
        langgraph.graph = graph
        sys.modules.setdefault("langgraph", langgraph)
        sys.modules.setdefault("langgraph.graph", graph)

    try:
        import langgraph.types  # noqa: F401
    except ModuleNotFoundError:
        types_module = types.ModuleType("langgraph.types")

        class Send:
            def __init__(self, node, arg):
                self.node = node
                self.arg = arg

            def __repr__(self):
                return f"Send({self.node!r}, {self.arg!r})"

        types_module.Send = Send
        sys.modules.setdefault("langgraph.types", types_module)


def install_logging_stub() -> None:
    """避免导入上游日志包装器时带入不相关运行时。"""
    if "core.execution_logger" in sys.modules:
        return
    module = types.ModuleType("core.execution_logger")
    module.log_error = lambda *args, **kwargs: None
    module.log_route = lambda *args, **kwargs: None
    module.log_tool_end = lambda *args, **kwargs: None
    module.log_tool_start = lambda *args, **kwargs: None
    sys.modules["core.execution_logger"] = module


def install_store_dependency_stub() -> None:
    """只替代 parent store 对 utils.clear_directory_contents 的依赖。"""
    if "utils" not in sys.modules:
        module = types.ModuleType("utils")

        def clear_directory_contents(directory):
            from pathlib import Path
            import shutil

            directory = Path(directory)
            if directory.is_dir():
                for child in directory.iterdir():
                    shutil.rmtree(child) if child.is_dir() else child.unlink()

        module.clear_directory_contents = clear_directory_contents
        sys.modules["utils"] = module
