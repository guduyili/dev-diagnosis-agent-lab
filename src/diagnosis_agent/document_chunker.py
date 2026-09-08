"""把 Markdown 文档转换成 parent/child 两级 Document。

父块保留较完整的章节上下文，子块较短、适合向量召回。流程先按标题切分，
再合并过小章节、拆分过大章节，最后修补边界过小的块；因此这里的长度是
字符数近似，不是 token 数。每一步都必须保留 metadata，才能从 child 回到
原文件和 parent。
"""

import os
import glob
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter



class DocumentChunker:
    """实现可配置的 Markdown 标题分块和父子块生成。"""

    def __init__(self):
        # 这些校验在启动时失败，避免运行到索引阶段才产生难以解释的碎片。
        if config.MIN_PARENT_SIZE <= 0 or config.MAX_PARENT_SIZE < config.MIN_PARENT_SIZE:
            raise ValueError("Parent chunk sizes must be positive and MIN_PARENT_SIZE <= MAX_PARENT_SIZE.")
        if not 0 <= config.CHILD_CHUNK_OVERLAP < config.CHILD_CHUNK_SIZE:
            raise ValueError("CHILD_CHUNK_OVERLAP must be smaller than CHILD_CHUNK_SIZE.")
        if config.CHILD_CHUNK_OVERLAP >= config.MAX_PARENT_SIZE:
            raise ValueError("CHILD_CHUNK_OVERLAP must be smaller than MAX_PARENT_SIZE.")


        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )

        # 递归字符文本分割器
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE
        self.__max_parent_size = config.MAX_PARENT_SIZE


    @staticmethod
    def __merge_metadata(target,source,prepend=False):
        """合并标题层级等 metadata，并去重保持原有顺序。

        同一个 parent 可能由多个标题块合并而来，因此 metadata 不能简单覆盖。
        使用 `` -> `` 连接层级后，检索结果仍能告诉模型内容来自哪些章节。
        """
        for key, value in source.items():
            # 不存在 则添加到target中
            if key not in target:
                target[key] = value
            # prepend 前置
            else:
                first,second = (value, target[key]) if prepend else(target[key],value)
                values = [
                    item.strip()
                    for raw in (first, second)
                    for item in str(raw).split(" -> ")
                    if item.strip()
                ]
                # 去重保持顺序
                target[key] = " -> ".join(dict.fromkeys(values))


    def create_chunks(self,path_dir=config.MARKDOWN_DIR):
        """处理目录下所有 Markdown，并返回所有 parent/child 列表。"""
        all_parent_chunks, all_child_chunks = [], []

        for doc_path_str in sorted(glob.glob(os.path.join(path_dir,"*.md"))):
            doc_path  = Path(doc_path_str)

            parent_chunks, child_chunks = self.__process_single_markdown(doc_path)



    def create_chunks_single(self, doc_path: Path):
        """处理单个 Markdown，并返回 parent/child 列表。"""
        return self.__process_single_markdown(doc_path)