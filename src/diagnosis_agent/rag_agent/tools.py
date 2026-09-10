"""将检索能力暴露给模型的工具工厂。

工具返回字符串是为了让模型容易阅读，但每个结果仍保留 parent_id 和 source。
评测或生产化时可进一步改成结构化 ToolResult，以区分空结果、工具错误和有效证据。
"""

from langchain_core.tools import tool
import config
from db.parent_store_manager import ParentStoreManager
from core.execution_logger import log_error, log_tool_end, log_tool_start

class ToolFactory:
    """把一个向量 collection 和 parent store 封装成两个 LangChain 工具。"""
    
    def __init__(self, collection):
        self.collection = collection
        # 此处新建的管理器与 RAGSystem.parent_store 默认指向相同目录，
        # 是两个对象访问同一批 JSON，不是复制一份 parent 内容到工具内部。
        self.parent_store_manager = ParentStoreManager()
    
    # 以下两个工具的英文 docstring 会成为发给模型的工具描述，因此学习说明
    # 放在 # 注释中。改 docstring 可能改变模型选工具的行为，不能视作纯文档变更。
    def _search_child_chunks(self, query: str, limit: int = config.DEFAULT_RETRIEVAL_K) -> str:
        """Search document excerpts for evidence related to the user question.

        Use this as the first retrieval step. Results include parent IDs, file
        names, and short child-chunk excerpts. If excerpts are relevant but too
        fragmented to answer confidently, call retrieve_parent_chunks with the
        returned parent_id.
        
        Args:
            query: Focused search query with concrete keywords from the question.
            limit: Maximum number of child chunks to return.
        """
        # 日志先记录请求；成功、空结果和异常都在对应分支记录结束，便于判断
        # “模型没有调用工具”与“工具调用了但没有召回”是两种不同故障。
        log_tool_start("search_child_chunks", {"query": query, "limit": limit})
        try:
            # k 是返回数量上限；阈值过滤后可以不足 k 或为零。这里返回 Document
            # 而不是 (Document, score)，输出文本没有保留检索分数供后续排序解释。
            results = self.collection.similarity_search(
                query,
                k=limit,
                score_threshold=config.RETRIEVAL_SCORE_THRESHOLD,
            )
            if not results:
                # 固定哨兵供模型阅读，也供 nodes._retrieval_contexts 排除空结果。
                # 空结果不等于库中肯定没有答案，还可能受 query、k、阈值影响。
                output = "NO_RELEVANT_CHUNKS"
                log_tool_end("search_child_chunks", output)
                return output

            # parent_id 是后续 retrieve_parent_chunks 的桥梁，不能从返回文本中删掉。
            output = config.CHILD_CHUNK_SEPARATOR.join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])
            log_tool_end("search_child_chunks", output)
            return output

        except Exception as e:
            # 转成字符串后，图会收到正常 ToolMessage 内容，而不是向外抛异常。
            # 因此“工具节点执行结束”不等于“成功获得证据”，评测需要识别此前缀。
            log_error("search_child_chunks", e)
            output = f"RETRIEVAL_ERROR: {str(e)}"
            log_tool_end("search_child_chunks", output)
            return output
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve the full parent chunk for a relevant child search result.

        Use this only after search_child_chunks returns a relevant parent_id and
        the child excerpt needs more surrounding context. Do not call this for
        parent IDs already available in compressed context.
    
        Args:
            parent_id: Parent chunk ID returned by search_child_chunks.
        """
        log_tool_start("retrieve_parent_chunks", {"parent_id": parent_id})
        try:
            # 按 ID 直接读本地文件，不做 embedding 或二次相似度检索。
            # 正常 load_content 返回非空字典；缺失文件会抛异常，通常进入 except，
            # 而不是下面的 NO_PARENT_DOCUMENT 分支。
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                output = "NO_PARENT_DOCUMENT"
                log_tool_end("retrieve_parent_chunks", output)
                return output

            output = (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )
            log_tool_end("retrieve_parent_chunks", output)
            return output

        except Exception as e:
            log_error("retrieve_parent_chunks", e)
            output = f"PARENT_RETRIEVAL_ERROR: {str(e)}"
            log_tool_end("retrieve_parent_chunks", output)
            return output
    
    def create_tools(self) -> list:
        """创建有稳定名称和签名的工具列表，交给 ``llm.bind_tools`` 和 ToolNode。"""
        # 绑定方法已固定 self，暴露给模型的参数只有 query/limit 或 parent_id。
        # bind_tools 告诉模型“可以请求什么”，ToolNode 才负责实际执行请求。
        # 英文描述中的首步检索、禁止重复读取属于提示词约束，此处没有强制检查。
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]
