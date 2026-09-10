"""D01 配置与协议；D02 运行真实图。先预测节点顺序，再看断言。"""
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from .support import UPSTREAM_PROJECT, ScriptedLLM


@pytest.mark.day(1)
def test_config_is_from_upstream_and_sizes_are_valid(upstream):
    config = upstream("config")
    assert Path(config.__file__).resolve() == (UPSTREAM_PROJECT / "config.py").resolve()
    assert 0 <= config.CHILD_CHUNK_OVERLAP < config.CHILD_CHUNK_SIZE
    assert 0 < config.MIN_PARENT_SIZE <= config.MAX_PARENT_SIZE
    assert config.MAX_TOOL_CALLS > 0 and config.MAX_ITERATIONS > 0
    # 构造真实分块器会运行上游的参数校验，但不加载 embedding。
    assert upstream("document_chunker").DocumentChunker() is not None


@pytest.mark.day(1)
def test_query_schema_requires_clarification_field(analysis):
    parsed = analysis(is_clear=True, questions=["response_format"], clarification_needed="")
    assert parsed.questions == ["response_format"]
    with pytest.raises(ValidationError):
        analysis(is_clear=True, questions=["response_format"])


@pytest.mark.day(2)
@pytest.mark.current_behavior
def test_real_graph_can_finish_without_tool_calls(graph_factory, analysis):
    # 观察当前边界：即使提示词要求首步检索，模型不发工具请求也能到聚合。
    # 不把该行为当成目标设计；若你增加硬性检索要求，应同步改这条契约。
    model = ScriptedLLM(
        analyses=[analysis(is_clear=True, questions=["response_format"], clarification_needed="")],
        decisions=[AIMessage(content="没有执行检索，无法核验。")],
        answers=[AIMessage(content="资料不足，待补充。")],
    )
    graph = graph_factory(model, [])
    cfg = {"configurable": {"thread_id": "day02"}, "recursion_limit": 50}
    events = list(graph.stream({"messages": [HumanMessage(content="response_format")]},
                               config=cfg, stream_mode="updates"))
    names = [name for event in events for name in event]
    assert names == ["summarize_history", "rewrite_query", "agent", "aggregate_answers"]
    state = graph.get_state(cfg)
    assert state.next == ()
    assert state.values["messages"][-1].content == "资料不足，待补充。"
    assert state.values["agent_answers"][0]["contexts"] == []
    assert [call["kind"] for call in model.calls] == ["analysis", "decision", "answer"]
    model.assert_consumed()


@pytest.mark.day(1)
def test_rag_system_wires_config_and_resets_checkpoint(upstream, settings, utilities, monkeypatch, tmp_path, parent_store):
    from .support import FixedDenseEmbeddings, FixedSparseEmbeddings
    vector_module = upstream("db.vector_db_manager")
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: FixedDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: FixedSparseEmbeddings())
    module = upstream("core.rag_system")
    tool_module = upstream("rag_agent.tools")
    monkeypatch.setattr(module, "ParentStoreManager", lambda: parent_store)
    monkeypatch.setattr(tool_module, "ParentStoreManager", lambda: parent_store)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(settings, "LLM_API_KEY", "offline-test-key")
    constructor_args = []
    model = ScriptedLLM()
    def build_model(**kwargs):
        constructor_args.append(kwargs)
        return model
    monkeypatch.setattr(module, "ChatOpenAI", build_model)
    system = module.RAGSystem()
    try:
        system.initialize()
        assert constructor_args[0]["model"] == settings.LLM_MODEL
        assert constructor_args[0]["base_url"] == settings.LLM_BASE_URL
        assert {tool.name for tool in model.bound_tools} == {"search_child_chunks", "retrieve_parent_chunks"}
        config = system.get_config()
        assert "callbacks" not in config
        assert config["recursion_limit"] == settings.GRAPH_RECURSION_LIMIT
        previous = system.thread_id
        system.agent_graph.update_state(config, {"pendingQuery": "old pending"})
        system.reset_thread()
        assert system.thread_id != previous
        assert system.agent_graph.get_state(config).values == {}
        assert system.agent_graph.get_state(system.get_config()).values == {}
    finally:
        system.vector_db._VectorDbManager__client.close()
