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

## [ERR-20260908-001] 参考项目虚拟环境缺少依赖和 pip

**Logged**: 2026-09-08
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
为注释修改执行导入验证时，参考项目 `.venv` 无法导入 `langchain_text_splitters`，且解释器没有 `pip` 模块。

### Error
```text
ModuleNotFoundError: No module named 'langchain_text_splitters'
No module named pip
```

### Context
- 操作：对 `project/` 执行 `compileall` 后导入核心模块。
- 解释器：`references/upstream/agentic-rag-for-dummies/.venv/Scripts/python.exe`。
- `compileall` 已通过；失败发生在需要第三方包的运行时导入阶段。
- 未记录密钥或完整环境变量。

### Suggested Fix
按 `requirements.txt` 重新创建或修复参考项目虚拟环境，并使用已验证的 Python 安装工具安装依赖；完成后重新做核心模块导入和最小离线实验。不要把本次静态编译结果写成完整运行通过。

### Metadata
- Reproducible: yes
- Related Files: `requirements.txt`, `docs/01-environment.md`
