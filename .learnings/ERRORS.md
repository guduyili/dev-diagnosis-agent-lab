# 错误记录

当前没有业务运行错误记录。

## [ERR-20260907-001] Python launcher 的过期登记

**Status**: resolved
**Area**: infra

现象：`py -0p` 列出了 Python 3.11，但 `py -3.11` 无法创建进程。
核查：登记的 Python311/python.exe 实际不存在；3.12 与 3.13 的绝对路径可以执行并返回版本。
处理：生成脚手架使用已验证的解释器；学习环境指南改为使用 Python 3.12 的已验证路径。
经验：解释器登记列表不等于运行验证，创建环境前实际执行 `--version` 并检查路径。

新增时记录：日期、任务、脱敏错误摘要、环境、最小复现、解决措施、验证结果。
环境错误与 Agent 业务失败分开分类；不要记录密钥、完整环境变量或无关用户文件。

## [ERR-20260907-002] DeepSeek 不接受 ChatOpenAI 默认的 structured output response_format

**Status**: resolved
**Area**: integration

现象：使用 DeepSeek V4 Flash 运行查询改写节点时，API 返回 `400 This response_format type is unavailable now`。
最小复现：`llm.with_structured_output(QueryAnalysis)` 在当前 LangChain/ChatOpenAI 组合下选择 provider-native `json_schema` 响应格式。
处理：在 `project/rag_agent/nodes.py` 显式指定 `method="function_calling"`，将结构化 schema 放入工具定义；DeepSeek 的工具调用接口支持该方式。
验证：源码检查确认运行时调用带有 `method="function_calling"`；通过 Python 编译检查和模块导入检查。
经验：切换 OpenAI 兼容模型时，不要假设所有 provider 都支持同一种 `response_format`。先确认模型能力，再显式选择 LangChain 的结构化输出方法。
