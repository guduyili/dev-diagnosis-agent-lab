"""负责把用户上传的文档转成 Markdown、分块并写入两个存储层。"""

from pathlib import Path
import shutil
import config
from utils import pdfs_to_markdowns, clear_directory_contents




class DocumentManager:
    """协调文件系统、分块器、parent JSON store 和 Qdrant。"""


    def __init__(self,rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)


    def add_documents(self, document_paths, progress_callback=None):
        """批量导入 PDF/Markdown，返回 ``(added, skipped)``。

        导入顺序是：复制/转换 → parent/child 分块 → 保存 parent → 写入 child
        向量。中途失败时尝试删除当前文件对应的 parent 和生成的 Markdown，尽量
        避免两层存储处于半成功状态；Qdrant 已写入的 child 仍需进一步清理，
        这是后续可改进的事务边界。
        """


        if not document_paths:
            return 0, 0
            
        # 单个字符串会包装成列表；单个 Path 对象没有同样的处理，调用者应传列表。
        # 不支持的扩展名在循环前被过滤，因此不会计入下面的 skipped。
        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md"]]

        if not document_paths:
            return 0, 0

        added = 0
        skipped = 0


        for i, doc_path in enumerate(document_paths):
            # 这是“开始处理第几个文件”，不是实际完成百分比：最后一项开始就报 1.0。
            # 回调位于 try 外；回调自身失败时不会进入下方的文件导入异常处理。

            if progress_callback:
                progress_callback((i + 1) / len(document_paths), f"Processing {Path(doc_path).name}")

            # 每个文件的处理顺序是：复制/转换 → parent/child 分块 → 保存 parent → 写入 child
            source_path = Path(doc_path)
            doc_name = source_path.stem
            md_path = self.markdown_dir / f"{doc_name}.md"
            
            # 只凭目标 Markdown 存在就跳过，不比较内容哈希，也不核对索引完整性。
            # a.pdf 和 a.md、不同目录的 a.md 都映射到同一个 a.md。
            # UI 的 Clear All 会清空整个知识库，并不是刷新单个文件的接口。
            if md_path.exists():
                skipped += 1
                continue

            # 每个文件重新初始化，失败清理不应删除前面成功导入的其他文件
            parent_ids = []
            try:
                if source_path.suffix.lower() == ".md":
                    shutil.copy(source_path, md_path)

                else:
                    # PDF 转 Markdown 时会丢弃图片，避免后续检索路径复杂化。
                    pdfs_to_markdowns(str(source_path), overwrite=False)
                parent_chunks,child_chunks = self.rag_system.chunker.create_chunks_single(
                    md_path,
                    source_name=source_path.name,
                )


                if not child_chunks:
                    raise ValueError("No child chunks were created.")

                parent_ids = [parent_id for parent_id, _ in parent_chunks]
                self.rag_system.parent_store.save_many(parent_chunks)
                
                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                # 此调用同时计算 child 向量写入库, 不是仅把文本登记到待处理队列
                collection.add_documents(child_chunks)

                print(f"✓ Imported {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks from {source_path.name}")
                # print(f" Collection '{self.rag_system.collection_name}' now has {collection.count()} child vectors.")

                added += 1 

            except Exception as e:
                # 补偿清理，不是事务回滚 没有清理可能已部分写入的child向量
                # delete_many/unlink 若再次抛错 会向外传播， 后面的文件也不会继续
                self.rag_system.parent_store.delete_many(parent_ids)
                if md_path.exists():
                    md_path.unlink()
                print(f"Error processing {doc_path}: {e}")
                skipped += 1
                
        # skipped 混合了“同名已存在”和“处理失败”，不能直接用它衡量导入质量。
        return added, skipped
            

    def get_markdown_files(self):
        # 优先使用 parent 来源标签；只要非空，就不再合并 Markdown 目录中的文件。
        # 两种展示来源都不能证明 child 索引完整，仅用于 UI 文件列表。
        sources = self.rag_system.parent_store.list_sources()

        if sources:
            return sources
        return sorted(p.name for p in self.markdown_dir.glob("*.md"))

    def clear_all(self):
        """删除当前 collection、Markdown 和 parent 文件，再创建空 collection。"""
        # 顺序操作、非原子操作；中间失败可能留下部分已删除的数据，没有自动恢复。
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.rag_system.vector_db.delete_collection(self.rag_system.collection_name)

        clear_directory_contents(self.markdown_dir)
        self.rag_system.parent_store.clear_store()

        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)