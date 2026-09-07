# 资源、来源与核查说明

核查日期：2026-09-07。核查范围为网页文档和目录说明；尚未在本项目安装、运行或验证依赖组合。

| 资源 | 用途 | 何时读 |
|---|---|---|
| [Agentic RAG 上游](https://github.com/GiovanniPasq/agentic-rag-for-dummies) | 参考实现和 Notebook | 第 1 周 |
| [上游 project 目录](https://github.com/GiovanniPasq/agentic-rag-for-dummies/tree/main/project) | 模块职责、运行与配置 | 第 1–4 周 |
| [上游 requirements](https://github.com/GiovanniPasq/agentic-rag-for-dummies/blob/main/requirements.txt) | 安装前核对所选版本 | 第 1 周 |
| [Hello-Agents](https://github.com/datawhalechina/hello-agents) | 第 4/6/8/9/12 章按需补原理 | 按问题阅读 |
| [LangGraph Academy](https://academy.langchain.com/courses/intro-to-langgraph) | 状态、路由、记忆 | 第 4 周 |
| [DeepEval 入门](https://deepeval.com/docs/getting-started) | 创建首个评测 | 第 5 周 |
| [DeepEval 工具正确性](https://deepeval.com/docs/metrics-tool-correctness) | 比较实际工具调用 | 第 5 周 |
| [DeepEval CI](https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd) | 回归门禁 | 第 7 周 |
| [Langfuse 数据集](https://langfuse.com/docs/evaluation/experiments/datasets) | 从真实执行沉淀用例 | 可选进阶 |
| [Playwright Test Agents](https://playwright.dev/docs/test-agents) | 测试开发方向扩展 | 第 8 周后 |
| [Schemathesis](https://github.com/schemathesis/schemathesis) | API 测试方向扩展 | 第 8 周后 |

## 使用方式

每次阅读前写一个问题，最多选一份主要资料；阅读后在本项目做一个可验证改动。
教学代码与当前框架 API 可能变化，按固定版本文档检查，不盲目复制旧课程代码。

## 本次确认的边界

- 上游参考应用偏向 Ollama，使用 Qdrant、父子分块与 LangGraph。换 provider 需要实际适配。
- 本项目没有自动接入 DeepEval，也没有设定裁判模型、凭据或云账户。
- DeepEval 的工具正确性默认检查和模型辅助模式需要区分。
- 学习计划中的任务数量、时间、数据划分与门禁都是建议，不是框架强制要求或招聘市场统计。
- 没有把上游 benchmark、Star 数或说明中的能力声明当成本项目验证结果。

## 来源记录习惯

代码：URL + 文件路径 + commit SHA + 许可 + 修改内容。
知识：出处 + 文档版本 + 获取日期 + 适用范围。
评测：用例来源 + 人工审核 + 版本 + 评分规则。
