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
