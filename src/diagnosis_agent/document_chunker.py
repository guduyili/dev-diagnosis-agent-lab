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
        去重并保持首次出现顺序，避免同一标题在不同层级重复出现。
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

            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
        return all_parent_chunks, all_child_chunks
        


    def create_chunks_single(self, md_path, source_name):
        """处理单个文件，返回 ``[(parent_id, parent), ...]`` 与 child 列表。

        parent_id 在最后一步按文件 stem 和顺序生成；这使同一输入和配置下的
        ID 可重复，但文件改名或分块配置变化会产生新的 ID。
        """
        doc_path = Path(md_path)
        source_name = source_name or f"{doc_path.stem}.pdf"


        with open(doc_path, "r", encoding="utf-8") as f:
            parent_chunks = self.__parent_splitter.split_text(f.read())


        # 标题切分得到的原始块可能太小；先合并才能让 parent 具有足够上下文。
        merged_parents = self.__merge_small_parents(parent_chunks)
        split_parents = self.__split_large_parents(merged_parents)
        cleaned_parents = self.__clean_small_chunks(split_parents)
        # 这里只强制检查上限。很短的整篇文档无法凭空补足内容，允许低于下限。
        if any(len(chunk.page_content) > self.__max_parent_size for chunk in cleaned_parents):
            raise ValueError("Parent chunking produced a chunk larger than MAX_PARENT_SIZE.")

        all_parent_pairs, all_child_chunks = [],[]

        self.__create_child_chunks(
            all_parent_pairs,
            all_child_chunks,
            cleaned_parents,
            doc_path,
            source_name,)
        return all_parent_pairs, all_child_chunks



    def __merge_small_parents(self, chunks):
        """从左到右合并章节，直到达到最小 parent 长度。"""
        if not chunks:
            return []

        merged, current = [], None
        # 遍历所有块，按顺序合并小块
        for chunk in chunks:
            if current is None:
                current = chunk
            # 如果当前块已经足够大，则直接加入 merged
            else:
                current.page_content += "\n\n" + chunk.page_content
                self.__merge_metadata(current.metadata, chunk.metadata, prepend=False)
            # 如果当前块已经达到最小长度，则将其加入 merged，并重置 current
            if len(current.page_content) >= self.__min_parent_size:
                    merged.append(current)
                    current = None

        # 处理最后一个块，如果它不够大，则与前一个块合并
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                self.__merge_metadata(merged[-1].metadata, current.metadata, prepend=False)
            else:
                merged.append(current)
        return merged

    def __split_large_parents(self, chunks):
        """将超过上限的 parent 拆成多个块，并保留原 metadata。"""
        split_chunks = []

        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:
                split_chunks.append(chunk)
            else:
                # 大块拆分使用child overlap,减少拆分点附近事实被截断的概率
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP,
                )
                # 拆分后每个子块都继承原块的 metadata
                sub_chunks = splitter.split_documents([chunk])
                # 重新合并 metadata，确保每个子块都包含原块的上下文信息
                split_chunks.extend(sub_chunks)
        return split_chunks 

    def __rebalance_pair(self, first, second):
        """重新切分相邻小块，使两边尽量满足 parent 的最小/最大长度。"""
        # 若相邻块来自带overlap的拆分， 拼接不会自动去掉重叠文本
        combined_content = first.page_content.rstrip() + "\n\n" + second.page_content.lstrip()

        # 令总长 N、上限 M、下限 m、切点 k：两块不超过 M 要求 N-M <= k <= M;
        # 1 <= k <= N-1 排除空块。仅 N >= 2m 时才有可能同时满足两块的下限。
        lower = max(len(combined_content) - self.__max_parent_size, 1)
        upper = min(self.__max_parent_size, len(combined_content) - 1)

        # 仅在总长大于两倍下限时才尝试平衡，否则直接返回原块。
        if len(combined_content) >= 2 * self.__min_parent_size:
            lower = max(lower, self.__min_parent_size)
            upper = min(upper,  len(combined_content) - self.__min_parent_size)
        # 首选切分点
        preferred = min(max(len(combined_content) //2, lower), upper)



        split_at = preferred
        # 按段落、换行、空格的优先级找边界；同类边界先选中点之前的一个，
        # 并不是比较前后距离取最近点。这里只识别字符分隔符，不解析代码语法。
        for separator in ("\n\n", "\n", " "):
            before = combined_content.rfind(separator, lower, preferred+1)
            after = combined_content.find(separator, preferred, upper+1)
            if before >= lower:
                split_at = before
                break
            if after != -1:
                split_at = after
                break

        left_text = combined_content[:split_at].rstrip()
        right_text = combined_content[split_at:].lstrip()
        # strip 可能让一侧低于下限，此时退回字符切点且保留空白以满足长度。
        if len(combined_content) >= 2 * self.__min_parent_size and (
            len(left_text) < self.__min_parent_size
            or len(right_text) < self.__min_parent_size
        ):
            split_at = preferred
            left_text,right_text = combined_content[:split_at], combined_content[split_at:]

        if not left_text or not right_text:
            return first, second
            

        # 两边各获得合并后的 metadata 副本；标签覆盖这对块的来源范围，
        # 不能解释成每一侧都精确包含全部标签对应的原文。
        metadata = dict(first.metadata)
        self.__merge_metadata(metadata, second.metadata)
        first.page_content,first.metadata = left_text,dict(metadata)
        second.page_content,second.metadata = right_text,dict(metadata)
        return first, second
    
    
    def __clean_small_chunks(self, chunks):
        """处理合并后仍过小的块，最后用相邻 pair 做一次再平衡。""" 
        cleaned = []
    

        # 第一轮优先并入前块，其次前置到后块；长度中的 +2 来自两个换行。
        # 前置到后块时不追加当前块，后续循环会处理被修改的 chunks[i + 1]。
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                # 合并在左边
                 if cleaned and len(cleaned[-1].page_content) + len(chunk.page_content) + 2 <= self.__max_parent_size:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    self.__merge_metadata(cleaned[-1].metadata, chunk.metadata, prepend=False)
                # 合并在右边
                 elif (
                     i < len(chunks) - 1
                     and len(chunk.page_content) + 2 + len(chunks[i + 1].page_content) <= self.__max_parent_size
                 ):
                     chunks[i+1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                     self.__merge_metadata(chunks[i+1].metadata, chunk.metadata, prepend=True) 


                 else:
                     cleaned.append(chunk)
            else:
                cleaned.append(chunk)


        # 第二轮尽量平衡遗留小块。只有一个块时直接保留，不承诺所有块都达到下限。
        for i, chunk in enumerate(cleaned):
            # 大于最小和者是最后一个块时不处理；否则尝试与下一个块平衡。
            if len(chunk.page_content) >= self.__min_parent_size or len(cleaned) == 1:
                continue
            
            if i < len(cleaned) -1:
                cleaned[i], cleaned[i + 1] = self.__rebalance_pair(cleaned[i], cleaned[i + 1])
            else:
                cleaned[i-1],cleaned[i] = self.__rebalance_pair(cleaned[i - 1], cleaned[i])

        return cleaned


    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path, source_name):
        """按文件 stem 和块序号生成 ID，再切成 child 并建立父子关联。"""
        # 例：guide.md -> guide_p0、guide_p1；不同目录的同名文件会发生 ID 冲突。
        # 此方法向调用者传入的两个列表追加结果，因此无需 return；它不负责持久化。
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_p{i}"
            # parent_id 和 source 同时写入 metadata，child 复制该 metadata，
            # 检索工具因此能返回“命中的文件 + 应回取的 parent”。
            p_chunk.metadata.update({
                "source": source_name,
                "parent_id": parent_id,})

            all_parent_pairs.append((parent_id, p_chunk))
            # 同一 parent 的多个 child 共享parent_id 它不是每一个 child的唯一主键
            all_child_chunks.extend(self.__child_splitter.split_documents([p_chunk]))
