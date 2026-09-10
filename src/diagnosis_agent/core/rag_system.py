"""组装向量库、聊天模型、检索工具和 LangGraph 图。

``RAGSystem`` 是基础设施组合根：它不实现具体节点，而是把各层对象
连接起来，并为当前实例维护一个可变的 thread_id。学习时可从 ``initialize``
向下追踪依赖，也可用假模型替换这里的 ChatOpenAI 来测试图的控制流。
"""

import uuid
from langchain_openai import ChatOpenAI
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChunker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability



class RAGSystem:
    """管理一次应用运行所需的 RAG 组件和会话检查点。"""

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        # collection 和 graph 在 initialize 中建立，但构造阶段已经有副作用：
        # VectorDbManager 加载 embedding 模型，ParentStoreManager 创建目录，
        # 启用的 Observability 可能联网认证。不能把 __init__ 当成纯配置赋值。
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChunker()
        self.observability = Observability()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        # UUID 区分检查点，不自动区分浏览器用户。当前 Gradio 回调共享本实例，
        # 因此也共享 thread_id；多用户隔离需要额外的每用户状态设计。
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    

    def initialize(self):
        """创建 collection、检查模型配置并编译主图。

        这里先检查 collection，再检查 DeepSeek key，最后才构建模型和图。
        这里只检查 key 非空，不验证 key 有效、模型可用或结构化输出协议兼容；
        构建 ChatOpenAI 和编译图成功，不等于已成功完成一次模型请求。
        """
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        if config.LLM_PROVIDER != "deepseek":
            raise ValueError(
                f"Unsupported LLM_PROVIDER={config.LLM_PROVIDER!r}; "
                "this fork is configured for DeepSeek's OpenAI-compatible API."
            )

        if not config.LLM_API_KEY:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy project/.env.example to "
                "project/.env and add your key before starting the app."
            )


        # DeepSeek 提供 OpenAI 兼容协议，因此复用 ChatOpenAI 适配器。
        # extra_body 请求服务端关闭思考模式，是否支持取决于服务端模型协议。
        # 它不是所有 response_format 400 错误的通用修复；rewrite 的结构化输出
        # 方法另见 nodes.py，学习调试时应区分模型参数和输出协议两个层面。
        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            # DeepSeek V4：关闭思考模式
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )
        # 此时得到的是工具对象列表，还没有绑定到聊天模型。create_agent_graph
        # 内部会调用 llm.bind_tools，并把相同工具交给 ToolNode 执行。
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def get_config(self):
        """返回 LangGraph invoke/stream 使用的线程和递归配置。"""
        # thread_id 用于定位检查点；recursion_limit 限制图调度步数，
        # 不是 Python 递归深度，也不是模型调用次数。它和子任务工具预算独立。
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        """清除当前内存检查点并生成新的会话 ID。

        当前使用 InMemorySaver，重启进程后本来就不会保留状态；这里的 reset
        只负责当前进程内隔离，不应被描述成持久化会话。
        """

        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        # 旧检查点删除失败仍切到新 ID，旧状态可能残留；这不会删除知识库文件。
        self.thread_id = str(uuid.uuid4())