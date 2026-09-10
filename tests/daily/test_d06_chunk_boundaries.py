"""D06 / W02-C：分块反例。每个测试中直接创建真实 DocumentChunker。

运行：python -m pytest tests/daily/test_d06_chunk_boundaries.py -v
自己改：增加“超长单行”“只有代码围栏”“无标题文档”三个输入之一。
current_behavior 记录当前局限，决定新契约后可以修改相应测试与上游实现。
"""
import pytest

import config
from document_chunker import DocumentChunker

pytestmark = pytest.mark.day(6)


def test_empty_directory_and_missing_file_have_different_results(tmp_path):
    chunker = DocumentChunker()
    assert chunker.create_chunks(tmp_path) == ([], [])
    with pytest.raises(FileNotFoundError):
        chunker.create_chunks_single(tmp_path / "absent.md")


@pytest.mark.parametrize("changes", [
    {"MIN_PARENT_SIZE": 0},
    {"MIN_PARENT_SIZE": 500, "MAX_PARENT_SIZE": 100},
    {"CHILD_CHUNK_OVERLAP": 160},
    {"CHILD_CHUNK_OVERLAP": -1},
])
def test_invalid_chunk_configuration_is_rejected(monkeypatch, changes):
    # 固定正常起点，再只改变表中的一组变量。
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)
    for name, value in changes.items():
        monkeypatch.setattr(config, name, value)
    with pytest.raises(ValueError):
        DocumentChunker()


@pytest.mark.current_behavior
def test_short_document_and_default_pdf_source_are_current_behavior(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    path = tmp_path / "short.md"
    path.write_text("# 短文\n很短", encoding="utf-8")
    chunker = DocumentChunker()

    parents, children = chunker.create_chunks_single(path)

    assert len(parents) == 1
    assert len(parents[0][1].page_content) < 80
    assert children[0].metadata["source"] == "short.pdf"  # 当前默认标签的局限。
    _, explicit_children = chunker.create_chunks_single(path, source_name=path.name)
    assert explicit_children[0].metadata["source"] == "short.md"
