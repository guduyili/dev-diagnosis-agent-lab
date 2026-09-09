"""D07–D09：真实本地索引、工具召回和生成输入，embedding/LLM 为已标明的替身。"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .support import DATA, ScriptedLLM, call_message, prompt_text

CASES = json.loads((DATA / "retrieval_cases.json").read_text(encoding="utf-8"))


@pytest.mark.day(7)
def test_parent_store_round_trip_overwrite_and_numeric_order(parent_store):
    parent_store.save("guide_p10", "十", {"source": "guide.md"})
    parent_store.save("guide_p2", "旧内容", {"source": "guide.md"})
    parent_store.save("guide_p2", "新内容", {"source": "guide.md"})
    records = parent_store.load_content_many(["guide_p10", "guide_p2", "guide_p2"])
    assert [row["parent_id"] for row in records] == ["guide_p2", "guide_p10"]
    assert records[0]["content"] == "新内容"
    assert parent_store.load("guide_p2.json")["page_content"] == "新内容"
    assert parent_store.list_sources() == ["guide.md"]


@pytest.mark.day(7)
def test_missing_and_corrupt_parent_are_explicit_errors(parent_store, settings):
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        parent_store.load("absent")
    (Path(settings.PARENT_STORE_PATH) / "broken.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        parent_store.load("broken")


@pytest.mark.day(7)
def test_real_qdrant_persists_and_reopens(library, upstream, settings):
    before = library.collection.similarity_search("response_format", k=1)
    assert before[0].metadata["source"] == "deepseek.md"
    # 关闭后重新构造真实 VectorDbManager，读取同一临时目录中的持久化数据。
    library.rag.vector_db._VectorDbManager__client.close()
    reopened = upstream("db.vector_db_manager").VectorDbManager()
    try:
        reopened.create_collection(settings.CHILD_COLLECTION)
        result = reopened.get_collection(settings.CHILD_COLLECTION).similarity_search("response_format", k=1)
        assert result[0].page_content == before[0].page_content
        assert result[0].metadata == before[0].metadata
    finally:
        reopened._VectorDbManager__client.close()


@pytest.mark.day(7)
def test_collection_rejects_incompatible_dimensions(vector_db, settings):
    from qdrant_client.http import models
    client = vector_db._VectorDbManager__client
    client.create_collection("wrong_size", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    with pytest.raises(ValueError, match="dense vector size"):
        vector_db.create_collection("wrong_size")


@pytest.mark.day(8)
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_actual_search_returns_expected_source_and_parent(tool_factory, parent_store, case):
    tools = {tool.name: tool for tool in tool_factory.create_tools()}
    result = tools["search_child_chunks"].invoke({"query": case["query"], "limit": 1})
    assert f"File Name: {case['expected_source']}" in result
    parent_id = result.split("Parent ID: ", 1)[1].splitlines()[0]
    parent = tools["retrieve_parent_chunks"].invoke({"parent_id": parent_id})
    assert case["expected_text"] in parent
    assert parent_store.load_content(parent_id)["content"] in parent


@pytest.mark.day(8)
def test_search_propagates_query_k_and_threshold(tool_factory, settings, monkeypatch):
    from unittest.mock import Mock
    spy = Mock(wraps=tool_factory.collection.similarity_search)
    monkeypatch.setattr(tool_factory.collection, "similarity_search", spy)
    result = tool_factory._search_child_chunks("parent_id", limit=2)
    spy.assert_called_once_with("parent_id", k=2, score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD)
    assert result.count("Parent ID: ") == 2


@pytest.mark.day(9)
def test_orchestrator_passes_tool_evidence_back_to_model(nodes):
    evidence = ToolMessage(content="File Name: deepseek.md\nContent: function_calling", tool_call_id="s1")
    state = {"question": "response_format", "messages": [
        HumanMessage(content="response_format", name="agent_question"),
        call_message("search_child_chunks", {"query": "response_format"}, "s1"), evidence,
    ]}
    model = ScriptedLLM(decisions=[AIMessage(content="使用 function_calling。")])
    update = nodes.orchestrator(state, model.bind_tools([]))
    assert evidence in model.calls[0]["messages"]
    assert update["iteration_count"] == 1 and update["tool_call_count"] == 0
    assert update["messages"][0].content == "使用 function_calling。"
    model.assert_consumed()


@pytest.mark.day(9)
def test_aggregate_sorts_subanswers_and_uses_original_question(nodes):
    model = ScriptedLLM(answers=[AIMessage(content="合并后的输出")])
    result = nodes.aggregate_answers({
        "messages": [], "originalQuery": "如何修复并验证？",
        "agent_answers": [
            {"index": 1, "answer": "SECOND_ANSWER", "contexts": ["RAW_CONTEXT_NOT_PASSED"]},
            {"index": 0, "answer": "FIRST_ANSWER", "contexts": []},
        ],
    }, model)
    prompt = prompt_text(model.calls[0])
    assert "如何修复并验证？" in prompt
    assert prompt.index("FIRST_ANSWER") < prompt.index("SECOND_ANSWER")
    assert "RAW_CONTEXT_NOT_PASSED" not in prompt
    assert result["messages"][-1].content == "合并后的输出"
    model.assert_consumed()


@pytest.mark.day(9)
def test_fallback_with_no_evidence_still_calls_model(nodes):
    model = ScriptedLLM(answers=[AIMessage(content="资料不足")])
    result = nodes.fallback_response({"question": "unknown", "messages": []}, model)
    assert "No data was retrieved" in prompt_text(model.calls[0])
    assert result["messages"][0].content == "资料不足"
    model.assert_consumed()
