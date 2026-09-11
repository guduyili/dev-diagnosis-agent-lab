"""D04 / W02-A：从 Markdown 到 parent JSON 和 child 向量的真实导入。

运行：python -m pytest tests/daily/test_d04_document_import.py -v
每个测试完整展示自己的准备过程；SimpleNamespace 只装入真实组件，不实现业务逻辑。
自己改：换文件内容、增加同名文件，观察“added/skipped”与实际存储之间的关系。
"""
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector

import config
import db.vector_db_manager as vector_module
from document_chunker import DocumentChunker
from core.document_manager import DocumentManager
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager

pytestmark = pytest.mark.day(4)

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


# 以下学习资料摘写自 nodes.py、document_chunker.py、parent_store_manager.py、graph.py。
# 400 错误文字来自用户报告；这些短资料不计入生产问题数据集。
DOCUMENTS = {
    "deepseek.md": (
        "# response_format 错误记录\n\n"
        "用户报告的实际错误文本是：This response_format type is unavailable now。\n"
        '本地 rewrite_query 使用 with_structured_output(QueryAnalysis, method="function_calling")。\n'
        "它将结构化结果放进工具参数，避免依赖 json_schema 响应格式。\n"
        "离线构造成功不能证明 DeepSeek 在线请求已经成功。\n"
    ),
    "parents.md": (
        "# parent_id 与父块存储\n\n"
        "DocumentChunker 使用文件 stem 和块序号生成 parent_id，例如 parents_p0。\n"
        "child 的 metadata 包含 parent_id 和 source。\n"
        "ParentStoreManager 将 page_content 和 metadata 保存为 JSON。\n"
        "相同 parent_id 再次保存会覆盖原文件；它不是内容哈希。\n"
    ),
    "sessions.md": (
        "# thread_id 与检查点\n\n"
        "create_agent_graph 使用 InMemorySaver 保存进程内状态。\n"
        "同一个 thread_id 用于查询、更新和恢复同一检查点。\n"
        "不同 thread_id 可保存不同的图状态。\n"
        "Gradio 当前共享一个 RAGSystem，不能因此声称页面已自动隔离不同用户。\n"
    ),
}


def test_markdown_import_records_actual_sources_and_parents(tmp_path, monkeypatch):
    # 1. 当天资料在本文件 DOCUMENTS 中；写入本次测试的上传目录。
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    paths = []
    for name, text in DOCUMENTS.items():
        path = upload_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))
    markdown_dir = tmp_path / "markdown"
    monkeypatch.setattr(config, "MARKDOWN_DIR", str(markdown_dir))
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())

    # 2. 逐个创建真实组件，再交给上游 DocumentManager。
    chunker = DocumentChunker()
    parent_store = ParentStoreManager(tmp_path / "parents")
    vector_db = VectorDbManager()
    try:
        vector_db.create_collection(config.CHILD_COLLECTION)
        components = SimpleNamespace(chunker=chunker, parent_store=parent_store,
                                     vector_db=vector_db, collection_name=config.CHILD_COLLECTION)
        manager = DocumentManager(components)

        # 3. 实际调用；三份文件都应成功导入。
        assert manager.add_documents(paths) == (3, 0)
        assert manager.get_markdown_files() == ["deepseek.md", "parents.md", "sessions.md"]

        listed_sources = parent_store.list_sources()
        print("\nListed sources:", listed_sources)
        stored = parent_store.load_content("deepseek_p0")
        assert stored["metadata"]["source"] == "deepseek.md"
        assert "This response_format type is unavailable now" in stored["content"]
        assert (markdown_dir / "deepseek.md").read_text(encoding="utf-8") == DOCUMENTS["deepseek.md"]
        result = vector_db.get_collection(config.CHILD_COLLECTION).similarity_search("response_format", k=1)
        assert result[0].metadata["source"] == "deepseek.md"
    finally:
        vector_db._VectorDbManager__client.close()


@pytest.mark.current_behavior
def test_reimport_skips_existing_filename_even_if_content_changed(tmp_path, monkeypatch):
    # 同一文件更新内容后再次导入，上游仍按名称跳过。
    path = tmp_path / "deepseek.md"
    path.write_text(DOCUMENTS["deepseek.md"], encoding="utf-8")
    monkeypatch.setattr(config, "MARKDOWN_DIR", str(tmp_path / "markdown"))
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    parent_store = ParentStoreManager(tmp_path / "parents")
    vector_db = VectorDbManager()
    try:
        vector_db.create_collection(config.CHILD_COLLECTION)
        manager = DocumentManager(SimpleNamespace(
            chunker=DocumentChunker(), parent_store=parent_store, vector_db=vector_db,
            collection_name=config.CHILD_COLLECTION,
        ))
        assert manager.add_documents([str(path)]) == (1, 0)

        path.write_text("# 新版本\n更新后内容", encoding="utf-8")
        assert manager.add_documents([str(path)]) == (0, 1)
        assert "更新后内容" not in parent_store.load_content("deepseek_p0")["content"]
    finally:
        vector_db._VectorDbManager__client.close()


def test_failed_empty_import_removes_generated_markdown(tmp_path, monkeypatch):
    path = tmp_path / "blank.md"
    path.write_text("", encoding="utf-8")
    markdown_dir = tmp_path / "markdown"
    parent_dir = tmp_path / "parents"
    monkeypatch.setattr(config, "MARKDOWN_DIR", str(markdown_dir))
    monkeypatch.setattr(config, "QDRANT_DB_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    monkeypatch.setattr(vector_module, "HuggingFaceEmbeddings", lambda **kw: LocalDenseEmbeddings())
    monkeypatch.setattr(vector_module, "FastEmbedSparse", lambda **kw: LocalSparseEmbeddings())
    parent_store = ParentStoreManager(parent_dir)
    vector_db = VectorDbManager()
    try:
        vector_db.create_collection(config.CHILD_COLLECTION)
        manager = DocumentManager(SimpleNamespace(
            chunker=DocumentChunker(), parent_store=parent_store, vector_db=vector_db,
            collection_name=config.CHILD_COLLECTION,
        ))

        assert manager.add_documents([str(path)]) == (0, 1)
        assert not (markdown_dir / "blank.md").exists()
        assert not (parent_dir / "blank_p0.json").exists()
    finally:
        vector_db._VectorDbManager__client.close()
