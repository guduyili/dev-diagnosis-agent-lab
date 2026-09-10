"""D13 / W05-A：真实工具结果如何变成评测上下文。

运行：python -m pytest tests/daily/test_d13_evaluation_context.py -v
自己改：加入一条错误或重复工具输出，先预测 contexts 的数量和内容。
这里不调用 DeepEval；它检验上游已有的上下文提取与子答案记录。
"""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, ToolMessage
from langchain_qdrant import SparseEmbeddings, SparseVector

import config
import db.vector_db_manager as vector_module
import rag_agent.tools as tools_module
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager
from rag_agent.tools import ToolFactory
from rag_agent.nodes import _retrieval_contexts, collect_answer

pytestmark = pytest.mark.day(13)

class LocalDenseEmbeddings(Embeddings):
    """只替代 embedding 模型；固定词表方便手算，不代表真实语义质量。"""
    def embed_query(self, text):
        terms = ("response_format", "parent_id", "thread_id", "pytest")
        return [float(text.lower().count(term)) for term in terms] + [0.01]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class LocalSparseEmbeddings(SparseEmbeddings):
    """固定稀疏向量；真实 Qdrant 负责写入、查询和融合。"""
    def embed_query(self, text):
        values = LocalDenseEmbeddings().embed_query(text)
        indices = [i for i, value in enumerate(values) if value > 0]
        return SparseVector(indices=indices, values=[values[i] for i in indices])

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def test_context_extraction_splits_actual_search_output_and_filters_errors(tmp_path, monkeypatch):
    # 本日检验工具输出转换，直接提供标准 child Document；不隐藏入库步骤。
    documents = [
        Document(page_content="parent_id 关联父块。", metadata={"source": "parents.md", "parent_id": "parents_p0"}),
        Document(page_content="JSON 通过 parent_id 保存 page_content。", metadata={"source": "store.md", "parent_id": "store_p0"}),
    ]
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    store = ParentStoreManager(tmp_path / "parents")
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: store)
    vector_db = VectorDbManager()
    try:
        vector_db.create_collection(config.CHILD_COLLECTION)
        collection = vector_db.get_collection(config.CHILD_COLLECTION)
        collection.add_documents(documents)
        factory = ToolFactory(collection)

        output = factory._search_child_chunks("parent_id", limit=2)
        messages = [
            ToolMessage(content=output, name="search_child_chunks", tool_call_id="s1"),
            ToolMessage(content=output, name="search_child_chunks", tool_call_id="s2"),
            ToolMessage(content="NO_RELEVANT_CHUNKS", tool_call_id="empty"),
            ToolMessage(content="RETRIEVAL_ERROR: broken", tool_call_id="err"),
            AIMessage(content="模型猜测，不是工具证据"),
        ]
        contexts = _retrieval_contexts(messages)

        assert contexts == output.split(config.CHILD_CHUNK_SEPARATOR)
        assert len(contexts) == 2
        assert all("File Name:" in value for value in contexts)
    finally:
        vector_db._VectorDbManager__client.close()


def test_collect_answer_keeps_actual_tool_contexts(tmp_path, monkeypatch):
    store = ParentStoreManager(tmp_path / "parents")
    store.save("deepseek_p0", "response_format：改用 function_calling。", {"source": "deepseek.md"})
    monkeypatch.setattr(tools_module, "ParentStoreManager", lambda: store)
    factory = ToolFactory(collection=None)  # parent 回取不依赖向量检索。
    output = factory._retrieve_parent_chunks("deepseek_p0")
    contexts = _retrieval_contexts([ToolMessage(content=output, tool_call_id="p1")])

    result = collect_answer({
        "messages": [AIMessage(content="实际传给 collect_answer 的输出")],
        "question": "response_format", "question_index": 2, "retrieved_contexts": contexts,
    })

    record = result["agent_answers"][0]
    assert record["index"] == 2
    assert record["answer"] == "实际传给 collect_answer 的输出"
    assert record["contexts"] == [output]
    assert "function_calling" in record["contexts"][0]
    assert record["question"] == "response_format"


@pytest.mark.parametrize("message", [
    AIMessage(content=""),
    AIMessage(content="", tool_calls=[{
        "name": "search_child_chunks", "args": {"query": "x"}, "id": "s1", "type": "tool_call",
    }]),
])
def test_collect_does_not_treat_empty_answer_or_tool_request_as_answer(message):
    result = collect_answer({"messages": [message], "question": "x", "question_index": 0})
    assert result["final_answer"] == "Unable to generate an answer."
