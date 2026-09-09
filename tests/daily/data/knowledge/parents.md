# parent_id 与父块存储

DocumentChunker 使用文件 stem 和块序号生成 parent_id，例如 parents_p0。
child 的 metadata 包含 parent_id 和 source。
ParentStoreManager 将 page_content 和 metadata 保存为 JSON。
相同 parent_id 再次保存会覆盖原文件；它不是内容哈希。
