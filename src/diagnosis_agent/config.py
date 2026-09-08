"""集中管理 Agentic RAG 的运行参数。

配置模块只负责把默认值和环境变量转换成 Python 常量；业务模块通过这里
读取路径、模型和预算，避免把参数散落在图节点或 UI 回调中。学习时修改
一个参数后，应记录它影响了哪条数据流，并重新建立索引或重新运行评测。
"""

import os

# --- Directory Configuration ---
# ``__file__`` 位于 project/config.py，因此向上两级得到仓库根目录。
# 所有相对存储都固定在这里，避免从不同工作目录启动 app 时写到不同位置。
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")

# --- Qdrant Configuration ---
# child chunk 放进 Qdrant；parent chunk 保存在 JSON 文件中，通过 parent_id 关联。
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
# embedding 模型决定向量索引的维度和语义；更换后必须检查并重建索引。
# 聊天模型使用 DeepSeek 的 OpenAI 兼容接口。API key 只从环境变量读取，
# 不把秘密写入源码或提交到 Git。
DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SPARSE_MODEL = "Qdrant/bm25"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek").strip().lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
JUDGE_MODEL = "ministral-3:3b-instruct-2512-q8_0"
LLM_TEMPERATURE = 0

# --- Retrieval Configuration ---
# score threshold 是检索过滤阈值，DEFAULT_RETRIEVAL_K 是每次最多返回的 child 数量。
# 分隔符用于把多个检索结果拼成模型可识别、又能在后处理中拆开的文本。
RETRIEVAL_SCORE_THRESHOLD = 0.4
DEFAULT_RETRIEVAL_K = 7
CHILD_CHUNK_SEPARATOR = "\n\n<CHILD_CHUNK_BOUNDARY>\n\n"

# --- Agent Configuration ---
# MAX_TOOL_CALLS 和 MAX_ITERATIONS 是安全边界，不是模型“应该”调用的次数。
# 图递归上限还包含普通节点跳转，通常应大于单个 Agent 的迭代上限。
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
MAIN_HISTORY_MESSAGES_TO_KEEP = 4
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Terminal Execution Logging ---
# execution_logger 只输出截断后的状态摘要，避免终端被长文档和完整上下文淹没。
EXECUTION_LOGGING_ENABLED = False
EXECUTION_LOG_MAX_CHARS = 1200
EXECUTION_LOG_USE_COLOR = True

# --- Text Splitter Configuration ---
# 先按 Markdown 标题形成 parent，再把 parent 切成适合向量召回的 child。
# parent 尺寸过小会缺上下文，过大会增加生成成本；应通过实验而不是凭感觉调整。
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
