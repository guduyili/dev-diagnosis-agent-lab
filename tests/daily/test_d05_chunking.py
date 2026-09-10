"""D05 / W02-B：直接观察真实 DocumentChunker 如何分块。

运行：python -m pytest tests/daily/test_d05_chunking.py -v
先读：project/document_chunker.py 的 create_chunks_single 和 create_chunks。
自己改：替换下方 Markdown、调整四个尺寸、增加一段超长代码，再观察哪个断言失败。
"""
import pytest

import config
from document_chunker import DocumentChunker

pytestmark = pytest.mark.day(5)


def test_chunks_preserve_code_error_and_parent_links(tmp_path, monkeypatch):
    # 1. 数据就在这里。错误文字来自用户报告；编号段落用于检测中间内容丢失。
    markdown = (
        "# response_format 故障\n\n"
        "FIRST_EVIDENCE\n\nERROR_400\n\n"
        "This response_format type is unavailable now\n\n"
        "```python\nmethod = 'function_calling'\n```\n\n"
    )
    markdown += "\n\n".join(f"CHECK_{i:02} 检查配置与日志。" for i in range(40))
    markdown += "\n\nLAST_EVIDENCE"

    # 2. 仅用临时文件；不写项目的 markdown_docs。
    path = tmp_path / "incident.md"
    path.write_text(markdown, encoding="utf-8")
    monkeypatch.setattr(config, "MIN_PARENT_SIZE", 80)
    monkeypatch.setattr(config, "MAX_PARENT_SIZE", 420)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 160)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 20)

    # 3. 直接创建上游类、调用公开函数。这里没有自定义 chunker fixture。
    chunker = DocumentChunker()
    parents, children = chunker.create_chunks_single(path, source_name=path.name)

    # 4. 结果既要有内容，也要能从 child 找到 parent。
    assert parents and children
    parent_map = dict(parents)
    assert len(parent_map) == len(parents), "parent ID 不应重复"
    parent_text = "\n".join(doc.page_content for _, doc in parents)
    child_text = "\n".join(doc.page_content for doc in children)
    expected = ["FIRST_EVIDENCE", "ERROR_400", "LAST_EVIDENCE", "function_calling"]
    expected += [f"CHECK_{i:02}" for i in range(40)]
    for text in expected:
        assert text in parent_text and text in child_text, f"内容丢失：{text}"
    for key, parent in parents:
        assert parent.metadata["parent_id"] == key
        assert parent.metadata["source"] == "incident.md"
        assert len(parent.page_content) <= 420
    for child in children:
        assert child.metadata["parent_id"] in parent_map
        assert child.metadata["source"] == "incident.md"
        assert 0 < len(child.page_content) <= 160


def test_directory_accumulates_all_files_and_returns_results(tmp_path):
    # 两份文件可抓住“循环执行了，但只返回最后一份结果”的错误。
    (tmp_path / "first.md").write_text("# 第一份\nfirst_evidence", encoding="utf-8")
    (tmp_path / "second.md").write_text("# 第二份\nsecond_evidence", encoding="utf-8")

    chunker = DocumentChunker()
    parents, children = chunker.create_chunks(tmp_path)

    assert {key for key, _ in parents} == {"first_p0", "second_p0"}
    assert {doc.metadata["parent_id"] for doc in children} == {"first_p0", "second_p0"}
    text = "\n".join(doc.page_content for _, doc in parents)
    assert "first_evidence" in text and "second_evidence" in text
