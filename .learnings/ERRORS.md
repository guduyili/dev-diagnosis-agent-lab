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

## [ERR-20260909-CMT] comment_review_paths

**Logged**: 2026-09-09
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary
注释复核时混用了任务工作目录与 G 盘上游目录，读取 CSS 和协作约定失败；一次日志补丁也因锚点不匹配被拒绝。

### Error
`Get-Content: Cannot find path`；`apply_patch verification failed: Failed to find expected lines`。

### Suggested Fix
源码读取明确指定上游 workdir；跨项目读取使用绝对路径。补丁必须使用已读到的真实行作为锚点。

### Resolution
已从正确目录读取 CSS 和主项目 AGENTS.md，并使用已核对的日志尾部追加记录。失败读取和失败补丁未改变源码。

另一次为减少换行提示而临时设置 `core.autocrlf=false`，导致现有 CRLF 被 diff 检查当成尾部空白。恢复仓库换行处理，并仅用 `core.safecrlf=false` 关闭转换提示后检查通过；未改动文件换行或放宽空白错误规则。

### Metadata
- Reproducible: yes
- Related Files: `references/upstream/agentic-rag-for-dummies/project/ui/css.py`, `AGENTS.md`

## [ERR-20260909-DAILY] daily_test_adaptation

**Logged**: 2026-09-09
**Priority**: medium
**Status**: resolved
**Area**: tests

### Summary
重写上游检验时发现 ToolNode 独立 invoke 缺运行时上下文，以及测试错误地期望首个 child 已包含 parent 才有的事实。

### Resolution
把 ToolNode 放进真实 StateGraph 执行；完整链路分别检查 child 的错误现象与 parent 回取后的修复方式，保留证据传递断言。59 项通过。

### Other setup corrections
初期猜测的 daily 文档路径不存在，改为读取实际的 10-implementation-workbook.md；补丁工具不支持同一补丁删除并新增相同路径，改为读取现有内容后 Update。依赖声明移除不存在的 pymupdf4llm[layout] extra，实际 pymupdf-layout 已由安装依赖提供。一次预判正则转义问题的补丁因行不存在被拒绝，复读确认原正则正确，未修改。

## [ERR-20260910-DAILY] test_file_rewrite_commands

**Logged**: 2026-09-10
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
生成测试补丁的 JavaScript 模板中未转义代码围栏反引号，导致脚本解析失败；一次 rg 文件参数也未适配 PowerShell 的通配符行为。

### Resolution
补丁通过普通字符串或安全转义传入，最终测试源码显示正常 Markdown 围栏；搜索改用 rg 的 -g 'test_d*.py' 过滤。失败工具调用未写入源文件。重写后的 59 项测试通过。
