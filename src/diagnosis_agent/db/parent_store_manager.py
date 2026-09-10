"""parent chunk 的轻量文件存储。

child chunk 进入向量库以便召回，完整 parent 保存在 JSON 中以便按 parent_id
回取上下文。这个设计简单、可检查，代价是缺少数据库事务和并发控制。
"""

import re
import json
import config
from utils import clear_directory_contents
from pathlib import Path
from typing import List, Dict


class ParentStoreManager:
    """用一个JOSN文件保存一个 parent chunk"""
    __store_path: Path

    def __init__(self, store_path=config.PARENT_STORE_PATH):
        self.__store_path = Path(store_path) 
        self.__store_path.mkdir(parents=True, exist_ok=True)


    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        # 磁盘协议是 {page_content, metadata}，与 load_content 返回的 content 键不同。
        # 同 ID 会覆盖旧文件；直接 write_text 不是临时文件替换，也没有批次事务。
        # 路径直接拼接 ID，当前方法没有验证 ID 是否含目录分隔符或越界路径。
        file_path = self.__store_path / f"{parent_id}.json"
        file_path.write_text(
            json.dumps({"page_content": content, "metadata": metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def save_many(self, parents: List) -> None:
        """批量保存 ``(parent_id, Document)`` 对。"""
        # 逐个写入：中途失败时，前面已经成功写入的文件仍然存在。
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)

    def delete_many(self, parent_ids: List[str]) -> None:
        for parent_id in parent_ids:
            file_path = self.__store_path / f"{parent_id}.json"
            if file_path.exists():
                file_path.unlink()


    def load(self, parent_id: str) -> Dict:
        """读取原始 JSON；不存在或损坏时让调用方显式处理异常。"""
        # 兼容传入带 .json 的名称。不返回 None：缺失抛 FileNotFoundError，
        # 格式损坏抛 JSONDecodeError；工具层决定如何把异常转成模型可读结果。
        file_path = self.__store_path /(
            parent_id if parent_id.lower().endwith(".json") else f"{parent_id}.json"
        )
        return json.loads(file_path.read_text(encoding="utf-8"))


    @staticmethod
    def _get_sort_key(id_str):
        # p2 应排在 p10 前，故提取数字而非字符串排序；旧格式 _parent_2 也兼容。
        # 不匹配的 ID 返回 0；本函数没有把文件 stem 纳入排序键。
#         _(?:parent_|p)(\d+)$
# │     │        │   │
# │     │        │   └─ 必须出现在字符串结尾
# │     │        └──── 捕获一个或多个数字
# │     └───────────── 匹配 parent_ 或 p
# └─────────────────── 前面必须有 _
        match = re.search(r'_(?:parent_|p)(\d+)$', id_str)
        return int(match.group(1)) if match else 0


    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        """去重并按 parent 序号排序；不同文件同序号之间的顺序未定义。"""
        # set 会丢失输入顺序，a_p0 与 b_p0 的排序键相同，不能保证跨进程顺序一致。
        # 列表推导中任一 load 失败，整个调用抛错，不会返回已成功读取的部分结果。
        unique_ids = set(parent_ids)
        return [self.load_content(pid) for pid in sorted(unique_ids, key=self._get_sort_key)]


    def list_sources(self) -> List[str]:
        """扫描 metadata.source，用于 UI 展示已导入的来源文件。"""
        # source 是存储的来源标签，不证明原文件仍在，也不证明向量库索引完整。
        # 这里只忽略读取和 JSON 语法错误，没有覆盖所有错误的 JSON 数据结构。
        sources = set()
        for file_path in self.__store_path.glob("*.json"):
            try:
                source = json.loads(file_path.read_text(encoding="utf-8")).get("metadata", {}).get("source")
                if source:
                    sources.add(source)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(sources)


    def clear_store(self) -> None:
        """清空 parent 文件，但保留存储目录本身。"""
        self.__store_path.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.__store_path)

