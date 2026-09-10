"""模型结构化输出的 Pydantic schema。"""

from typing import List
from pydantic import BaseModel, Field

# 该类交给 with_structured_output(QueryAnalysis, method="function_calling")：
# 模型按 schema 提交函数参数，LangChain 解析为 QueryAnalysis 对象。
# 因而节点可以写 response.questions，而不是自行对回答字符串 json.loads。
# 类 docstring 和 Field(description=...) 可能进入模型可见的 schema；
# 学习解释写在 # 注释里，修改描述字符串则可能改变提示词行为。
class QueryAnalysis(BaseModel):
    """查询改写节点的协议：清晰度、可检索问题和澄清提示。"""
    # 必填字段；没有默认值。这个布尔值表示“问题可检索”，不表示已找到证据。
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )
    # 上面的 List[str] 不限制列表长度，也不保证字符串非空。
    # “最多 3 个问题”来自 Prompt，Python/Pydantic 在这里没有对应 max_length。
    # 澄清文本也必填，但可为空字符串；可用性判断位于 rewrite_query。
