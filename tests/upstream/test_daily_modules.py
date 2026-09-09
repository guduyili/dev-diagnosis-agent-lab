"""按每日路线验证上游 agentic-rag-for-dummies 的关键模块。

运行：
    .\\.venv\\Scripts\\python.exe -m pytest tests/upstream -q

这些测试只使用临时 Markdown、内存替身和本地 reducer，不启动 Gradio、Qdrant、
embedding 下载或 DeepSeek。真实 provider 协议和端到端质量仍需单独人工/集成验收。
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from .support import (
    install_langgraph_stubs,
    install_logging_stub,
    install_store_dependency_stub,
)


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PROJECT = ROOT / "references" / "upstream" / "agentic-rag-for-dummies" / "project"


@pytest.mark.upstream_day(1)
def test_upstream_python_sources_parse_without_importing_the_app():
    files = sorted(UPSTREAM_PROJECT.rglob("*.py"))
    assert files
    for path in files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


@pytest.mark.upstream_day(1)
def test_upstream_config_keeps_chunk_and_budget_invariants():
    config = importlib.import_module("config")
    assert 0 <= config.CHILD_CHUNK_OVERLAP < config.CHILD_CHUNK_SIZE
    assert 0 < config.MIN_PARENT_SIZE <= config.MAX_PARENT_SIZE
    assert config.MAX_TOOL_CALLS > 0
    assert config.MAX_ITERATIONS > 0


@pytest.mark.upstream_day(6)
def test_document_chunker_processes_a_directory_and_keeps_parent_links(tmp_path, monkeypatch):
    config = importlib.import_module("config")
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 20)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 30)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 5)

    chunker_module = importlib.import_module("document_chunker")
    chunker = chunker_module.DocumentChunker()
    (tmp_path / "guide.md").write_text(
        "# Upload\n\nfirst_evidence\n\n## Error\n\nsecond_evidence",
        encoding="utf-8",
    )
    parents, children = chunker.create_chunks(tmp_path)

    assert parents and children
    parent_ids = {parent_id for parent_id, _ in parents}
    assert {doc.metadata["parent_id"] for _, doc in parents} == parent_ids
    assert {doc.metadata["parent_id"] for doc in children} <= parent_ids
    all_parent_text = "\n".join(doc.page_content for _, doc in parents)
    assert "first_evidence" in all_parent_text
    assert "second_evidence" in all_parent_text


@pytest.mark.upstream_day(6)
def test_document_chunker_empty_directory_is_an_explicit_empty_result(tmp_path):
    chunker = importlib.import_module("document_chunker").DocumentChunker()
    assert chunker.create_chunks(tmp_path) == ([], [])


@pytest.mark.upstream_day(7)
def test_parent_store_round_trip_and_numeric_order(tmp_path):
    install_store_dependency_stub()
    manager = importlib.import_module("db.parent_store_manager").ParentStoreManager(tmp_path)
    manager.save("guide_p10", "ten", {"source": "guide.md"})
    manager.save("guide_p2", "two", {"source": "guide.md"})
    manager.save("guide_p0", "zero", {"source": "guide.md"})

    loaded = manager.load_content_many(["guide_p10", "guide_p2", "guide_p2", "guide_p0"])
    assert [item["parent_id"] for item in loaded] == ["guide_p0", "guide_p2", "guide_p10"]
    assert loaded[1]["content"] == "two"
    assert manager.list_sources() == ["guide.md"]


@pytest.mark.upstream_day(7)
def test_parent_store_missing_file_is_not_silently_treated_as_empty(tmp_path):
    install_store_dependency_stub()
    manager = importlib.import_module("db.parent_store_manager").ParentStoreManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.load_content("does-not-exist")


@pytest.mark.upstream_day(8)
def test_graph_state_reducers_have_explicit_append_union_and_reset_semantics():
    install_langgraph_stubs()
    state = importlib.import_module("rag_agent.graph_state")
    assert state.append_unique(["a", "b"], ["b", "c"]) == ["a", "b", "c"]
    assert state.set_union({"search::upload"}, {"search::upload", "parent::p0"}) == {
        "search::upload", "parent::p0"
    }
    assert state.accumulate_or_reset([{"answer": "old"}], [{"answer": "new"}]) == [
        {"answer": "old"}, {"answer": "new"}
    ]
    assert state.accumulate_or_reset([{"answer": "old"}], [{"__reset__": True}]) == []


@pytest.mark.upstream_day(10)
@pytest.mark.parametrize(
    ("clear", "questions", "expected"),
    [(True, ["upload"], "agent"), (True, [], "agent"), (False, [], "request_clarification")],
)
def test_query_route_uses_upstream_clarity_flag(clear, questions, expected):
    install_langgraph_stubs()
    install_logging_stub()
    edges = importlib.import_module("rag_agent.edges")
    result = edges.route_after_rewrite({"questionIsClear": clear, "rewrittenQuestions": questions})
    if expected == "request_clarification":
        assert result == expected
    else:
        # The upstream route creates one Send per rewritten question. For an empty
        # list the implementation returns an empty fan-out, which this test records.
        assert expected == "agent"
        assert isinstance(result, list)
        assert len(result) == len(questions)


@pytest.mark.upstream_day(10)
def test_orchestrator_route_stops_a_tool_batch_when_budget_is_exceeded():
    install_langgraph_stubs()
    install_logging_stub()
    edges = importlib.import_module("rag_agent.edges")

    class Message:
        tool_calls = [{"name": "search_child_chunks", "args": {}}]

    result = edges.route_after_orchestrator_call(
        {"messages": [Message()], "iteration_count": edges.MAX_ITERATIONS, "tool_call_count": 0}
    )
    assert result == "fallback_response"


@pytest.mark.upstream_day(11)
def test_tool_factory_returns_explicit_empty_and_error_results():
    install_store_dependency_stub()
    install_logging_stub()
    tools = importlib.import_module("rag_agent.tools")

    class EmptyCollection:
        def similarity_search(self, *args, **kwargs):
            return []

    factory = tools.ToolFactory(EmptyCollection())
    assert factory._search_child_chunks("missing") == "NO_RELEVANT_CHUNKS"

    class BrokenCollection:
        def similarity_search(self, *args, **kwargs):
            raise RuntimeError("fixture failure")

    broken = tools.ToolFactory(BrokenCollection())
    assert broken._search_child_chunks("missing").startswith("RETRIEVAL_ERROR:")


@pytest.mark.upstream_day(11)
def test_query_analysis_schema_requires_all_protocol_fields():
    schema = importlib.import_module("rag_agent.schemas").QueryAnalysis
    result = schema(is_clear=True, questions=["upload"], clarification_needed="")
    assert result.is_clear is True
    assert result.questions == ["upload"]
    with pytest.raises(Exception):
        schema(is_clear=True, questions=["upload"])
