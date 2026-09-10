"""D04–D06：真实 Markdown 入库、分块和失败边界，数据只写 tmp_path。"""
from pathlib import Path

import pytest

from .support import DATA


@pytest.mark.day(4)
def test_markdown_import_records_actual_sources_and_parents(library, settings):
    assert library.manager.get_markdown_files() == ["deepseek.md", "parents.md", "sessions.md"]
    stored = library.rag.parent_store.load_content("deepseek_p0")
    assert stored["metadata"]["source"] == "deepseek.md"
    assert "This response_format type is unavailable now" in stored["content"]
    assert (Path(settings.MARKDOWN_DIR) / "deepseek.md").read_text(encoding="utf-8") == (
        DATA / "knowledge/deepseek.md").read_text(encoding="utf-8")


@pytest.mark.day(4)
@pytest.mark.current_behavior
def test_reimport_skips_existing_filename_even_if_content_changed(library, tmp_path):
    # 上游按文件名跳过，不检查内容哈希；改进版本更新机制前先认识现状。
    newer = tmp_path / "deepseek.md"
    newer.write_text("# 新版本\n更新后内容", encoding="utf-8")
    assert library.manager.add_documents([str(newer)]) == (0, 1)
    assert "更新后内容" not in library.rag.parent_store.load_content("deepseek_p0")["content"]


@pytest.mark.day(4)
def test_failed_empty_import_removes_generated_markdown(library, settings, tmp_path):
    empty = tmp_path / "blank.md"
    empty.write_text("", encoding="utf-8")
    assert library.manager.add_documents([str(empty)]) == (0, 1)
    assert not (Path(settings.MARKDOWN_DIR) / "blank.md").exists()
    assert not (Path(settings.PARENT_STORE_PATH) / "blank_p0.json").exists()


@pytest.mark.day(5)
def test_chunks_preserve_code_error_and_parent_links(chunker, tmp_path, settings):
    path = tmp_path / "incident.md"
    # 明确构造边界资料，编号让中间段丢失也能被检测；不是虚构业务事故。
    lines = ["# 故障记录", "FIRST_EVIDENCE", "ERROR_400",
             "```python\nmethod = 'function_calling'\n```"] + [
        f"CHECK_{i:02} 检查配置与日志。" for i in range(40)
    ] + ["LAST_EVIDENCE"]
    path.write_text("\n\n".join(lines), encoding="utf-8")
    parents, children = chunker.create_chunks_single(path, source_name=path.name)
    assert parents and children
    parent_map = dict(parents)
    assert len(parent_map) == len(parents)
    parent_text = "\n".join(doc.page_content for _, doc in parents)
    child_text = "\n".join(doc.page_content for doc in children)
    for token in ["FIRST_EVIDENCE", "ERROR_400", "LAST_EVIDENCE", "function_calling"] + [
        f"CHECK_{i:02}" for i in range(40)
    ]:
        assert token in parent_text and token in child_text, f"丢失内容：{token}"
    for key, parent in parents:
        assert parent.metadata["parent_id"] == key
        assert parent.metadata["source"] == path.name
        assert len(parent.page_content) <= settings.MAX_PARENT_SIZE
    for child in children:
        assert child.metadata["parent_id"] in parent_map
        assert child.metadata["source"] == path.name
        assert 0 < len(child.page_content) <= settings.CHILD_CHUNK_SIZE


@pytest.mark.day(5)
def test_directory_accumulates_all_files_and_returns_results(chunker, tmp_path):
    for name in ("first", "second"):
        (tmp_path / f"{name}.md").write_text(f"# {name}\n{name}_evidence", encoding="utf-8")
    parents, children = chunker.create_chunks(tmp_path)
    assert {key for key, _ in parents} == {"first_p0", "second_p0"}
    assert {doc.metadata["parent_id"] for doc in children} == {"first_p0", "second_p0"}
    assert "second_evidence" in "\n".join(doc.page_content for _, doc in parents)


@pytest.mark.day(6)
def test_empty_directory_and_missing_file_have_different_results(chunker, tmp_path):
    assert chunker.create_chunks(tmp_path) == ([], [])
    with pytest.raises(FileNotFoundError):
        chunker.create_chunks_single(tmp_path / "absent.md")


@pytest.mark.day(6)
@pytest.mark.parametrize("changes", [
    {"MIN_PARENT_SIZE": 0},
    {"MIN_PARENT_SIZE": 500, "MAX_PARENT_SIZE": 100},
    {"CHILD_CHUNK_OVERLAP": 160},
    {"CHILD_CHUNK_OVERLAP": -1},
])
def test_invalid_chunk_configuration_is_rejected(upstream, settings, monkeypatch, changes):
    for key, value in changes.items():
        monkeypatch.setattr(settings, key, value)
    with pytest.raises(ValueError):
        upstream("document_chunker").DocumentChunker()


@pytest.mark.day(6)
@pytest.mark.current_behavior
def test_short_document_and_default_pdf_source_are_current_behavior(chunker, tmp_path):
    path = tmp_path / "short.md"
    path.write_text("# 短文\n很短", encoding="utf-8")
    parents, children = chunker.create_chunks_single(path)
    assert len(parents) == 1
    assert len(parents[0][1].page_content) < 80
    assert children[0].metadata["source"] == "short.pdf"
    # 显式 source_name 可正确保留 md 标签；DocumentManager 就是这样调用的。
    explicit = chunker.create_chunks_single(path, source_name=path.name)
    assert explicit[1][0].metadata["source"] == "short.md"
