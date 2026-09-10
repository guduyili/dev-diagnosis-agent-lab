"""可选 Langfuse tracing 的最小封装。"""

import logging
import config

logger = logging.getLogger(__name__)


class Observability:
    """按配置初始化 callback，并提供显式 flush 方法。"""

    def __init__(self):
        self._enabled = config.LANGFUSE_ENABLED
        self._handler = None
        self._client = None

        # 默认关闭时完全不导入 langfuse，保持本地学习路径轻量。
        if not self._enabled:
            return

        if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
            logger.warning("Langfuse enabled but API keys are missing — skipping")
            self._enabled = False
            return

        try:
            # 延迟导入让没有安装 Langfuse 的环境仍可运行核心 RAG。
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler

            # 未把 config 中的 key/host 作为参数传入，客户端依赖 SDK 自己读取环境。
            # 因而“config 中定义了某个变量”不等于 SDK 一定使用了该配置值。
            self._client = get_client()

            # 真实认证请求发生在构造阶段；失败时禁用追踪，核心 RAG 仍可继续初始化。
            if self._client.auth_check():
                print("Langfuse client is authenticated and ready!")
            else:
                print("Authentication failed. Please check your credentials and host.")
                self._enabled = False
                return

            self._handler = CallbackHandler()
        except Exception as exc:
            logger.warning("Could not initialize Langfuse: %s", exc)
            self._enabled = False

    def get_handler(self):
        """返回可传给 LangGraph config.callbacks 的 handler，未启用时为 None。"""
        return self._handler

    def flush(self):
        """尽力提交缓冲 trace；flush 失败不应阻断用户回答。"""
        # 当前由 ChatInterface.clear_session 调用，不是每轮 chat 结束自动调用。
        # 仅检查 client 是否存在；即使初始化后半段失败，仍可能尝试 flush。
        # 异常被忽略，方法返回不代表服务器已成功收齐所有追踪事件。
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
