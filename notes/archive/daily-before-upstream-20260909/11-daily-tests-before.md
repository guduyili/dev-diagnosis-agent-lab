# 每日任务完成后的测试与人工验收

创建日期：2026-09-09。配套 [学习工作手册](10-implementation-workbook.md)。

本次新增 63 个 pytest 自动检查，覆盖 17 个学习时段；另 7 个时段以人工证据验收。D01～D24 是“学习时段”的顺序编号，每周对应 A/B/C 三项任务，不要求连续 24 个自然日完成。每个时段可拆成多天。

测试目标是你自己的 `src/diagnosis_agent/`，不会自动替换成上游的完整代码。当前已经存在部分分块练习，其他接口还待实现，因此最初运行会出现失败；通过实现满足约定，让对应阶段逐步通过。

## 1. 每天如何运行

主项目根目录：`G:\Learning\dev-diagnosis-agent-lab`。

本次已在主项目登记并安装开发依赖 pytest、langchain-text-splitters，版本由 `uv.lock` 固定。后续在新环境先执行：

```powershell
Set-Location 'G:\Learning\dev-diagnosis-agent-lab'
uv sync --frozen
$env:PYTHONUTF8 = "1"

# 第一次先看能收集哪些检查，不执行业务：
.\.venv\Scripts\python.exe -m pytest tests/daily --collect-only -q

# 只检查今天：D06 与 W02-C 等价
.\.venv\Scripts\python.exe -m pytest tests/daily --day D06 -q

# 报错太多时先处理第一个
.\.venv\Scripts\python.exe -m pytest tests/daily --day D06 -x -q

# 今日通过后，再检查前面所有自动任务
.\.venv\Scripts\python.exe -m pytest tests/daily --through-day D06 -q

# 单独验证一条有意义的行为
.\.venv\Scripts\python.exe -m pytest tests/daily/test_w02_knowledge.py -k directory -q
```

必须在命令中保留 `tests/daily`，使 pytest 在解析参数时加载该目录的学习日插件。也可以使用 [快捷脚本](../scripts/check_day.ps1)：

```powershell
.\scripts\check_day.ps1 -Day D06
.\scripts\check_day.ps1 -Day W02-C -Cumulative
```

快捷脚本使用主项目虚拟环境，并在 `reports/raw/` 输出按时间命名的 JUnit XML，保留 pytest 的退出码。如果 PowerShell 当前不允许执行脚本，直接使用上面的 Python 命令，无需修改全局执行策略。

不要运行上游目录的 Python 来验收自己的主项目，也不要把上游目录加入 PYTHONPATH 来掩盖包内导入错误。

## 2. 结果怎么理解

| 输出 | 含义 | 下一步 |
|---|---|---|
| passed | 本次选择的这一项自动行为检查通过 | 检查人工要求，再运行累计检查 |
| NOT_IMPLEMENTED | 业务模块或约定的入口还不存在 | 按对应时段实现；不能写恒定返回值骗过用例 |
| IMPORT_ERROR | 模块存在，但内部导入有问题或缺第三方依赖 | 看缺失模块名，区分包路径问题与安装问题 |
| BEHAVIOR_OR_RUNTIME_FAILURE | 断言不满足或业务执行抛出其他异常 | 从输入和预期行为定位，不删除断言 |
| NETWORK_BLOCKED | 本测试进程尝试建立 Python TCP 连接 | 注入假模型/embedding/工具，勿在离线验收中调真实 API |
| deselected | 没有选中的未来或其他学习日 | 没运行，不计为通过 |
| 人工验收提示、退出码 4 | 该时段需人工证据，或参数错误 | 按人工清单处理，不算自动通过 |

缺少实现时使用失败而不是 skip/xfail；这会让 pytest 的失败或错误计数增加。发生在 fixture 初始化时显示 ERROR，发生在测试体内显示 FAILED；请结合分类提示判断原因。

退出码 0 表示选中的自动测试通过；1 表示有失败/错误；4 表示参数或人工验收提示；5 表示未收集到测试。全部测试通过也不能证明你能独立讲解、在线模型可用或产品达到生产质量。

测试阻断本进程常用 Python TCP 连接；这是防止意外联网的辅助措施，不是对任意未知程序的安全沙箱。测试不运行外部命令、不读真实密钥、不初始化在线裁判。

## 3. 24 个学习时段对应表

| 学习日 | 手册任务 | 自动文件或人工检查 | 完成判据 |
|---|---|---|---|
| D01 / W01-A | 环境与参数 | `test_w01_start.py` | 主项目源码可解析；分块、top-k、循环参数自洽 |
| D02 / W01-B | 真实请求追踪 | 人工 | 正常、澄清、无资料三条路径；输入输出与实际节点轨迹 |
| D03 / W01-C | 收集问题 | 人工 | 真实 incident、来源、复现状态；没有虚构成功 |
| D04 / W02-A | 数据字段 | `test_w02_knowledge.py` | 来源与版本可追溯；必需字段为空时拒绝 |
| D05 / W02-B | 加载器 | 同上 | MD/TXT 中文正确读取；同名路径和修订 ID 正确 |
| D06 / W02-C | 分块与边界 | 同上 | 全目录有返回；文本保留；parent/child 关联正确 |
| D07 / W03-A | 索引 | `test_w03_rag.py` | 幂等写入、更新、持久化和 embedding 兼容检查 |
| D08 / W03-B | 检索 | 同上 | 项目/版本过滤、top-k 和已知相关项排序 |
| D09 / W03-C | 普通 RAG | 同上 | 单次检索、真实上下文传递、无证据与工具异常分支 |
| D10 / W04-A | 状态与路由 | `test_w04_agent.py` | 增量正确合并；清晰度与问题列表共同决定路由 |
| D11 / W04-B | 工具边界 | 同上 | 整批预算检查；请求、执行、成功数量分别记录 |
| D12 / W04-C | 会话 | 同上 | 两会话隔离；澄清恢复；解决后清空 pending |
| D13 / W05-A | 用例加载 | `test_w05_evaluation.py` | 拒绝空集/重复/未审核/坏数据，检测家族跨集合 |
| D14 / W05-B | 评测适配 | 同上 | 参考答案不泄漏；actual_output 和 context 来自真实结果 |
| D15 / W05-C | 裁判校准 | 人工 | 正确、错误、部分正确、应澄清对照；人工复核评分 |
| D16 / W06-A | JUnit 解析 | `test_w06_reports.py` | 正常、failure、error、skipped 分类和双根节点格式 |
| D17 / W06-B | 报告工具 | 同上 | 实际解析数据与来源；坏报告/越界路径明确报错 |
| D18 / W06-C | 实际使用 | 人工 | 五次真实任务记录及人工修正量，不能靠 mock 证明提效 |
| D19 / W07-A | 回归统计 | `test_w07_w08_regression.py` | 五种结果都计数；分母明确；全错误没有有效成功率 |
| D20 / W07-B | 门禁 | 同上 | 关键错误、跳过、未运行和空集不得放行 |
| D21 / W07-C | 红→绿演练 | 人工 | 对真实业务改动注入一个错误，捕捉失败并恢复 |
| D22 / W08-A | 版本对照 | `test_w07_w08_regression.py` | 按 case_id 比较，列退化项，拒绝不一致的数据集 |
| D23 / W08-B | 实验与展示 | 人工 | 演示一个成功、一个失败、一次回归；解释局限 |
| D24 / W08-C | 独立解释 | 人工 | 个人贡献、许可与下轮计划；脱离源码说明设计 |

表中是对手册 A/B/C 的验收拆分，个别相邻任务重新分配到更适合测试的接口；不替代手册中的全部学习产物。例如 D09 自动检查数据流，你仍要做 top-k 的实际对照实验。

## 4. 现有代码怎么接测试

已写好的分块器直接使用你现有的公开接口：

```python
from diagnosis_agent.document_chunker import DocumentChunker

chunker = DocumentChunker()
parents, children = chunker.create_chunks_single(path, source_name=path.name)
parents, children = chunker.create_chunks(directory)
```

parent 是 `(parent_id, Document)` 对，child 是含 `page_content` 和 `metadata` 的对象。测试把配置缩小以制造分块边界，再检查结果，不逐字模仿私有函数。

其他尚未实现的部分使用下述最小约定。它们是本轮补充的验收接口，不是宣称已有业务能力，也不是框架官方 API。可用 dataclass、Pydantic 或 dict 表示字段；已有自己的接口时，可写薄适配层，但适配层必须真正调用你的业务代码。

### D04：Document

入口 `diagnosis_agent.models.Document`，可用关键字构造：

- document_id、project_id、project_version、source_path、content_hash、text。
- document_id / project_version / source_path 为空抛 ValueError。
- 这些数据是来源契约；不要求你的类继承某个特定框架。

### D05：load_document

入口 `diagnosis_agent.knowledge.loader.load_document(path, *, root, project_id, project_version)`：

- 返回上述 Document 字段，source_path 可用相对路径或完整路径。
- 文档 ID 标识项目内路径；相同路径的修订仍用同一 document_id，content_hash 随文本更新。
- 不同目录里的同名文件不能共用 document_id。
- 缺文件抛 FileNotFoundError；纯空白文档抛 ValueError。

如果你选择其他修订策略，先写明决策，再调整测试契约；不能让“更新检测”的行为要求消失。

### D07～D08：索引

入口 `diagnosis_agent.retrieval.index.create_index(index_dir, *, embedder, embedding_id)` 返回对象：

- `upsert(chunks)`：接收测试 `support.chunk()` 中的字段，以 chunk_id 更新记录。
- `search(query, *, k, project_id, project_version)`：返回有来源字段的结果列表。
- `close()`：释放存储资源，之后同目录可重新打开。
- 相同 ID 重复写入不增加条数；相同 ID 的新内容替换旧内容。
- 持久化 embedding_id，不兼容时抛 ValueError。
- k 小于 1 抛 ValueError；不存在的项目返回空列表。

`TinyEmbeddings` 只提供固定向量；测试没有实现检索器。你仍需编写真正的索引、过滤和排序。此处不需要下载 Qwen 模型。

### D09：普通 RAG

入口 `diagnosis_agent.baseline.diagnose(request, *, retriever, llm)`：

- request 包含 question、project_id、project_version、session_id。
- retriever 提供 `search(query, **kwargs)`；传递项目和版本过滤。
- llm 提供 `generate(question, contexts)`，contexts 是实际检索片段的字符串列表。
- 返回 status、answer、evidence；错误分支还需 stop_reason。
- 本练习约定空证据直接返回 insufficient_evidence；检索异常返回 tool_error；有效结果回答状态为 answered。

假模型只记录调用和返回固定文本；不会帮被测代码传递过滤参数、拼装证据或选择失败分支。

### D10～D12：Agent

- `diagnosis_agent.agent.state.apply_update(previous, update)`：计数按增量累加；retrieved_contexts 去重追加；其他字段覆盖；不修改 previous。
- `diagnosis_agent.agent.nodes.route_query(analysis)`：当 is_clear 为真且 questions 非空返回 retrieve，否则 clarify。
- `diagnosis_agent.tools.executor.execute_tools(calls, *, registry, remaining)`：calls 为 name/args 列表。先检查整批预算，再调用 registry 中的实际函数。
- 工具返回 status（ok/tool_error/budget_exceeded）、proposed、executed、succeeded、tool_calls；单条记录保留实际 output 或 error_code。超过预算整批不执行，未知工具不执行，函数抛错算执行过但不成功。
- `diagnosis_agent.service.create_service(*, llm, retriever)`：返回具有 `diagnose(request)` 的服务实例，按 session_id 管理 pending query。
- 测试 LLM 除 generate 外提供 `analyze(question)`，返回 QueryAnalysis 三个字段。服务负责准备自包含 question、保存/恢复会话并调用业务流程。

这组离线测试未验证真实 provider 的工具协议、硬超时和跨进程恢复。完成这些能力时需要另加集成测试；不能只看这里变绿就声称全部具备。

### D13～D14：评测输入

本轮约定业务评测代码未来放在主项目根目录 `evals/`；目前的 `docs/evals/` 只是说明文档，没有把它移动或当作实现。

- `evals.cases.load_cases(path, *, source_root)`：返回 dict 列表，按已有用例约定校验，所有格式/审核/来源异常抛 ValueError；source_refs 相对 source_root 解析。
- `evals.cases.validate_splits({"dev": [...], "holdout": [...]})`：同一 family_id 跨集合抛 ValueError。
- `evals.adapter.run_case(case, *, diagnose)`：仅将 case["input"] 交给 diagnose；得到真实 result 后再转换。
- 转换输出字段：input、actual_output、expected_output、retrieval_context、tools_called。
- result 的 generation_contexts 是实际提供给生成器的上下文；不能用 expected 或事后补齐资料代替。
- 本练习使用 expected["summary"] 作为 expected_output。格式可按未来 rubric 扩展，但业务输入与答案必须隔离。
- 业务执行抛出 RuntimeError 时保留错误，不构造空答案冒充有效样本。

adapter 可返回 dict 或 DeepEval LLMTestCase；这组测试只核验字段，当前未安装或调用 DeepEval 裁判。工具名、参数及结果的更严格映射可在正式接入 ToolCall 后继续增加。

### D16～D17：报告解析

- `diagnosis_agent.tools.junit_report.parse_junit(path)` 返回含 cases 列表的结果。
- 每条 case 具有 id、status、message、duration；状态为 passed/failed/error/skipped。
- id 要包含类等上下文以区分重名测试；message 保留断言正文；duration 为数值。
- 无测试项、坏 XML、未知根结构抛 ValueError；缺文件抛 FileNotFoundError。
- `read_report_tool(path, *, allowed_root)` 返回 status、data、source_path、executed_tests。
- 成功状态为 ok，executed_tests 必须为 False；它只读取报告。
- 超出允许目录返回 status=error、error_code=PATH_OUTSIDE_ROOT；解析失败返回 INVALID_REPORT。

### D19～D22：统计、门禁与版本对照

`evals.reporting` 未来提供：

- `summarize(rows)`：rows 为 case_id/status 的列表；空集、重复 case_id 或未知状态抛 ValueError。
- 返回 total、passed、failed、error、skipped、not_run 六种计数。
- completion_rate=(passed+failed)/total；success_rate_all=passed/total。
- success_rate_judged=passed/(passed+failed)，分母为 0 时为 None。
- `gate(rows)`：本轮严格策略要求非空且全部 passed；其他情况 False。
- `compare_runs(baseline, candidate, *, baseline_meta, candidate_meta)`：按 case_id 对齐，返回 improved、regressed 的 ID 列表。
- meta 包含 dataset_hash 和 knowledge_hash；此轮固定资料做模型/代码对照，数据集或用例集合不一致时拒绝比较。若以后实验刻意改变知识库，需另定义该实验的比较约定。

D20 检验门禁函数，不代表已经配置 CI 或把结果接到进程退出码；后者在 D21 用实际运行验证。

## 5. 当前第一次运行结果

2026-09-09 在主项目环境运行：

- 收集到 63 项自动测试。
- D01：3 passed，60 deselected。
- D06：1 failed、4 errors，全部由 `import config` 的包路径问题触发；没有执行到分块算法断言。
- 全量：3 passed、53 failed、7 errors；未用 skip/xfail 隐藏未实现项。
- 全量未通过的 60 项中，55 项标为 NOT_IMPLEMENTED，5 项标为 IMPORT_ERROR。
- 原始报告：`reports/raw/daily-initial-2026-09-09.xml`，属于本机检查记录，不是 RAG 效果评测。

你的 `document_chunker.py` 在包内仍写 `import config`。按 `diagnosis_agent.document_chunker` 导入时，应先解决包内配置导入。测试不会把源码目录塞进 sys.path 来掩盖问题。

源码阅读还发现 `__merge_mall_parents` 与定义的 `__merge_small_parents` 拼写不同，以及若干函数尚未返回完整结果。这些是后续排查线索，目前并没有执行到相应分支。本轮保留练习代码，供你自行修复。

## 6. 一个测试怎么帮助你学习

以目录分块测试为例：

```python
parents, children = chunker.create_chunks(tmp_path)
assert parents and children
text = "\n".join(parent.page_content for _, parent in parents)
assert "first_evidence" in text and "second_evidence" in text
```

它检验三个可观察行为：公开函数返回约定结构、目录中两份文件都被处理、重要内容保留。仅在循环里调用单文件函数而忘记汇总/return，会使测试失败。

操作顺序：

1. 先读测试名、输入和断言，用自然语言说出行为。
2. 预测自己代码会返回什么，再运行单条测试。
3. 从第一个错误开始修，不同时修改几个层次。
4. 单条通过后运行当天全部，再运行累计检查。
5. 用不同于 fixture 的一份自己的小资料再检查，避免只适配示例。
6. 日记记录命令、实际 passed/failed/error、失败原因和新增理解；只有自动检查和人工要求都满足，才标记学习任务完成。

测试不是完整规格：例如 ID 碰撞、复杂 PDF、并发写入、恶意 XML、长会话和在线裁判误判均需要后续针对性案例。发现实际失败后将最小复现加入相关学习日。

## 7. 人工验收记录模板

七个人工时段不提供“文件存在就算学会”的自动测试。沿用 notes/daily 的记录方式，填以下内容：

```text
学习日：
对应手册章节：
实际命令与输入：
观察到的输出/轨迹：
证据文件位置：
我原来的预测：
出现差异的原因：
尚未验证的内容：
不看源码能否解释：
下一步：
```

D15 需附人工评分和裁判分歧；D18 需实际任务及修正记录；D21 需失败与修复后两次输出；D23/D24 需演示、贡献与设计解释。不要将构造测试数据直接计入真实问题集。

## 8. 文件入口

- [测试选择与离线保护](../tests/daily/conftest.py)
- [公共替身与业务入口加载](../tests/daily/support.py)
- [D01 基础检查](../tests/daily/test_w01_start.py)
- [D04～D06 知识与分块](../tests/daily/test_w02_knowledge.py)
- [D07～D09 索引与基线](../tests/daily/test_w03_rag.py)
- [D10～D12 Agent](../tests/daily/test_w04_agent.py)
- [D13～D14 评测适配](../tests/daily/test_w05_evaluation.py)
- [D16～D17 报告解析](../tests/daily/test_w06_reports.py)
- [D19、D20、D22 回归](../tests/daily/test_w07_w08_regression.py)

## 9. 上游参考项目模块测试

如果当天学习的是上游 `agentic-rag-for-dummies`，可以运行独立的上游验收，不会导入主项目的 `src/diagnosis_agent`：

```powershell
Set-Location 'G:\Learning\dev-diagnosis-agent-lab'
.\.venv\Scripts\python.exe -m pytest tests/upstream -q
```

`tests/upstream/test_daily_modules.py` 使用真实上游函数，输入使用临时 Markdown、临时 JSON 和内存替身。13 项检查对应以下学习内容：

| 上游学习日 | 自动检查 |
|---|---|
| D01 | 上游源码可解析，分块和预算配置自洽 |
| D06 | `DocumentChunker` 目录处理、内容保留、parent/child 关联、空目录 |
| D07 | `ParentStoreManager` 写入、回读、数字序号排序、缺失文件异常 |
| D10 | `graph_state` 的追加、并集、reset，以及查询澄清路由 |
| D10 | 子 Agent 达到迭代预算时转入 fallback |
| D11 | `ToolFactory` 的空检索和检索异常结果；`QueryAnalysis` 必填协议 |

这组测试会用小型 `MessagesState`、`Send` 和日志替身隔离未安装的 LangGraph 运行时，但被测的 reducer、路由和工具方法仍来自上游源码。它不证明 Qdrant、embedding、Gradio、DeepSeek 或 Langfuse 已经运行；这些需要在上游环境装好依赖后的集成测试和人工轨迹中验证。

主项目测试与上游测试分开运行：前者检查你正在实现的 `src/diagnosis_agent` 契约，后者检查参考实现的阅读结果。不要把两组测试混用，也不要通过把上游目录加入主项目包路径来掩盖主项目的未实现项。
