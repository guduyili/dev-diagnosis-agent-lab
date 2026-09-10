"""只替换模型服务边界；业务分块、检索、路由、图和存储全部来自上游。

ScriptedLLM 记录真实节点输入并返回安排的响应；不实现备用 Agent 算法。
"""
from collections import deque
from copy import deepcopy
from pathlib import Path
from threading import Lock

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PROJECT = ROOT / "references/upstream/agentic-rag-for-dummies/project"
DATA = Path(__file__).with_name("data")


class FixedDenseEmbeddings(Embeddings):
    """固定词表测试向量，不具有 Qwen 的语义质量；检索仍由真实 Qdrant 完成。"""
    terms = ("response_format", "parent_id", "thread_id", "pytest")

    def embed_query(self, text):
        return [float(text.lower().count(term)) for term in self.terms] + [0.01]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class FixedSparseEmbeddings(SparseEmbeddings):
    """替代 BM25 模型加载，不替代向量写入、查询或融合适配器。"""
    def embed_query(self, text):
        dense = FixedDenseEmbeddings().embed_query(text)
        indices = [i for i, value in enumerate(dense) if value > 0]
        return SparseVector(indices=indices, values=[dense[i] for i in indices])

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class ScriptedLLM:
    """控制外部响应来验证真实节点和控制流；不用于评价模型答题能力。"""
    def __init__(self, *, analyses=(), decisions=(), answers=()):
        self.queues = {"analysis": deque(analyses), "decision": deque(decisions),
                       "answer": deque(answers)}
        self.calls = []
        self.bound_tools = []
        self.structured_options = []
        self._lock = Lock()

    def _invoke(self, kind, messages):
        with self._lock:
            self.calls.append({"kind": kind, "messages": deepcopy(messages)})
            assert self.queues[kind], f"非预期的 {kind} 模型调用；检查图路径/循环次数"
            response = self.queues[kind].popleft()
        if isinstance(response, Exception):
            raise response
        return deepcopy(response(messages) if callable(response) else response)

    def invoke(self, messages, **kwargs):
        return self._invoke("answer", messages)

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _ModelEndpoint(self, "decision")

    def with_structured_output(self, schema, **kwargs):
        self.structured_options.append((schema, kwargs))
        return _ModelEndpoint(self, "analysis")

    def assert_consumed(self):
        assert not any(self.queues.values()), "安排的模型响应未用完：实际路径与预期不同"


class _ModelEndpoint:
    def __init__(self, owner, kind):
        self.owner, self.kind = owner, kind

    def invoke(self, messages, **kwargs):
        return self.owner._invoke(self.kind, messages)


def call_message(name, args, call_id="call-1"):
    """生成标准 LangChain AIMessage，交给真实路由和 ToolNode 消费。"""
    from langchain_core.messages import AIMessage
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": call_id, "type": "tool_call",
    }])


def prompt_text(call):
    return "\n".join(str(message.content) for message in call["messages"])
