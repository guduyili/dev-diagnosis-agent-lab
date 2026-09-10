# 每日模块实验：数据与调用过程写在当天文件中

更新：2026-09-10。按用户要求重写为 **一个自动学习日一个 test_dXX 文件**。共有 18 个文件、59 项测试；其余 6 个时段需要人工证据。

测试直接导入本地上游：

G:\Learning\dev-diagnosis-agent-lab\references\upstream\agentic-rag-for-dummies\project

每个文件依次展示：真实模块 import → 当天数据 → 临时文件或状态 → 实际函数调用 → 结果断言 → 可自行修改的方向。学习者无需追到 support.py 或业务 fixture 才知道对象怎么来的。

## 1. 先打开并运行一个文件

建议先看 [D05 分块实验](../tests/daily/test_d05_chunking.py)。其中的数据直接写在 markdown 变量中，DocumentChunker 也在测试函数内创建。

~~~powershell
Set-Location 'G:\Learning\dev-diagnosis-agent-lab'
# 新环境先安装已锁定的依赖；已有环境无需每天安装。
uv sync --frozen
$env:PYTHONUTF8 = '1'

# 只运行分块文件，不需要先执行 D01～D04。
.\.venv\Scripts\python.exe -m pytest tests/daily/test_d05_chunking.py -v

# 只运行其中一项；失败后停下查看原因。
.\.venv\Scripts\python.exe -m pytest tests/daily/test_d05_chunking.py -k parent_links -x -v

# 原有按日命令继续可用。
.\.venv\Scripts\python.exe -m pytest tests/daily --day D05 -v
.\scripts\check_day.ps1 -Day D12 -Cumulative

# 全量回归。
.\.venv\Scripts\python.exe -m pytest tests/daily -q
~~~

“独立运行”指直接用 pytest 指定该文件，且不依赖先运行别的学习日；不是直接执行 python test_d05_chunking.py，后者不会自动运行 pytest 测试函数。

如果断言失败，可在单条命令中加 --pdb，查看真实上游函数的变量。所有脚本从主项目根目录运行，使用主项目 .venv。

## 2. 现在的结构

| 内容 | 放在哪里 |
|---|---|
| 上游模块的 import | 当前 test_dXX 文件头部 |
| Markdown、问题、消息和期望数据 | 当前测试函数，或当前文件清晰列出的 DOCUMENTS |
| DocumentChunker、ParentStoreManager、VectorDbManager、ToolFactory 的创建 | 使用它们的测试函数内部 |
| 分块、入库、调用工具、运行图 | 在当前函数中逐步展开 |
| 模型响应 | 当前函数内的 Mock.return_value / Mock.side_effect |
| 离线 embedding | 当前文件中 LocalDenseEmbeddings / LocalSparseEmbeddings，代码完整可见 |
| 导入路径、--day 选择、阻止意外联网 | conftest.py；不创建业务对象 |
| 人工学习任务 | 本指南后面的人工清单 |

已移除运行中的 support.py、旧 test_wXX 文件和独立 data 文件。保留少量重复代码，是为了让学习者在一个文件中看懂实验全过程。

前一版文件与数据备份在 notes/archive/daily-fixtures-20260910/，Python 文件以 .py.txt 保存，不参与测试收集。更早的历史备份继续保留。此次没有修改上游业务实现或你正在编写的 src 代码。

## 3. 一个分块实验的读法

~~~python
from document_chunker import DocumentChunker

def test_example(tmp_path):
    # 输入
    markdown = "# 错误记录\n\nresponse_format 错误需要核查 function_calling。"
    path = tmp_path / "incident.md"
    path.write_text(markdown, encoding="utf-8")

    # 调用真实上游代码
    chunker = DocumentChunker()
    parents, children = chunker.create_chunks_single(path, source_name=path.name)

    # 检查实际输出
    assert parents and children
    assert children[0].metadata["parent_id"] in dict(parents)
    assert children[0].metadata["source"] == "incident.md"
~~~

这是展示阅读顺序的小例子；正式 D05 测试还检查字符上限、编号段落和关键内容保留。不要用这个小例子替换完整断言。

每天按下面顺序做：

1. 读文件顶部“运行、先读、自己改”说明。
2. 找到输入数据，预测函数的输出和异常。
3. 运行单条，查看实际结果。
4. 在当前文件里改一个输入或参数，预测变化后重跑。
5. 再修改上游源码，先跑当天，最后跑累计回归。
6. 写下能解释的行为、失败原因和下一条自己的反例。

## 4. 按学习日程打开哪个文件

| 学习日 | 文件（相对于 tests/daily） | 真实被测内容 | 可自行增加的实验 |
|---|---|---|---|
| D01 / W01-A | test_d01_config_and_system.py | config、DocumentChunker 校验、QueryAnalysis、RAGSystem 组装/reset | 先做前两项，再读较完整的系统组装 |
| D02 / W01-B | test_d02_graph_flow.py | create_agent_graph 的节点顺序、实际检查点与模型调用次数 | 给模型增加工具请求 |
| D04 / W02-A | test_d04_document_import.py | DocumentManager 从 MD 到 JSON/Qdrant；同名和空文件 | 内容更新、来源标签 |
| D05 / W02-B | test_d05_chunking.py | 单文件/目录分块、内容和父子关联 | 长代码、无标题、异常栈 |
| D06 / W02-C | test_d06_chunk_boundaries.py | 缺文件、空目录、非法尺寸、短文 | 选择并修复一种分块边界 |
| D07 / W03-A | test_d07_stores.py | parent 写回/覆盖/排序、Qdrant 持久化、维度检查 | 同序号跨文件、索引模型版本 |
| D08 / W03-B | test_d08_retrieval.py | 实际分块/建库、ToolFactory 搜索、按真实返回 ID 回取 | 新增 query 和人工期望来源 |
| D09 / W03-C | test_d09_generation.py | orchestrator、aggregate_answers、fallback_response 的输入输出 | 检验工具上下文缺失的情况 |
| D10 / W04-A | test_d10_state_and_routes.py | 真实 reducer、StateGraph、Send、路由 | 增量从 1 改为 2、重复数据 |
| D11 / W04-B | test_d11_tools_and_budget.py | 工具 schema、真实 ToolNode、预算边界、错误转换 | 坏 parent JSON、未知工具 |
| D12 / W04-C | test_d12_clarification.py | 真实暂停恢复、显式双 thread、SDK 请求体 | 多次澄清或坏服务端响应 |
| D13 / W05-A | test_d13_evaluation_context.py | 真实工具输出 → _retrieval_contexts / collect_answer | 重复/错误消息过滤 |
| D14 / W05-B | test_d14_compression.py | compress_context、summarize_history、删除消息与保留证据 | 改压缩阈值和历史保留数 |
| D16 / W06-A | test_d16_logging.py | logged_node 返回/异常、token 估算 | 注入不同异常、长日志 |
| D17 / W06-B | test_d17_ui_messages.py | 工具卡片 ID、展示截断、JSON 缓冲、关闭追踪 | 同名工具的多个调用 ID |
| D19 / W07-A | test_d19_full_agent.py | 完整导入→搜索→parent→子答案→汇总，以及跨轮 reset | 加自己的缺陷回归 |
| D20 / W07-B | test_d20_failure_regression.py | 超预算整批不执行、模型错误从真实图传播 | 将预算从 1 改成 2 |
| D22 / W08-A | test_d22_top_k_experiment.py | 同一真实索引上仅改变 k | 加同关键词无关资料 |

D03：真实 incident 与来源审核。
D15：真实接入裁判后的人工校准。
D18：实际任务的使用时间与修正量。
D21：对真实源码进行一次红→绿回归。
D23：展示成功、失败与回归各一个。
D24：独立讲解、个人贡献、许可和后续计划。

单独选择这些人工日会给出说明并返回 pytest 退出码 4，不会用“文件存在”冒充学习完成。D16/D17 使用上游现有日志/UI 作为原路线的前置实验，原计划的 JUnit 解析器仍需要你之后实现。

## 5. 这些测试辅助工具各做什么

| 工具 | 用途 | 没有做什么 |
|---|---|---|
| tmp_path | pytest 为当前测试分配临时目录 | 不预先替你分块或建立知识库 |
| monkeypatch | 在当前测试设置参数/替换一个外部边界，结束后恢复 | 不修改上游源码文件 |
| capsys | 捕获终端输出，用于日志断言 | 不代替日志实现 |
| Mock.invoke.side_effect = [响应1, 响应2] | 第一次、第二次调用分别返回什么；多调会失败 | 不实现 Agent 路由 |
| Mock.call_args / call_args_list | 查看真实节点传给模型/依赖的参数 | 不生成业务输入 |
| Mock(wraps=真实方法) | 记录调用，同时执行原方法 | 不用固定值代替检索 |
| SimpleNamespace | 把已创建的真实组件交给 DocumentManager | 不实现分块、存储或检索 |
| LocalDense/SparseEmbeddings | 将可见的固定词表变成向量，避免下载模型 | 不替换 Qdrant 查询/融合算法 |

D11 的检索失败用例在函数内明确注入异常，只检查真实 ToolFactory 如何处理异常；正常检索由 D08/D19 的实际本地库验证。

需要模型的测试全部在当前函数列出返回数据。D12 的协议测试使用真实 ChatOpenAI/OpenAI SDK，只在 HTTP 服务端边界返回测试 JSON，不向 DeepSeek 发请求。token 实验显式选择上游已有的 len//4 分支，不把它当精确计费。

## 6. 怎么按自己的进度改

- 学模块：先改本文件的 Markdown、状态、query 或预设模型响应，再解释输出。
- 学改进：先声明目标行为、增加能失败的反例，再改上游源码使测试通过。
- 学独立复刻：复制相应实验到 tests/unit，显式把 import 改到自己的模块；别在导入失败时偷偷回退到上游。
- 学真实效果：换真实 embedding/模型后另建集成或评测入口，明确凭据和费用；不要把离线固定响应算成真实问答成功。

current_behavior 标记记录当前局限：首步检索未强制、同名文件跳过更新、默认 .pdf 来源标签、失败 parent 请求仍被记录。它们不是要求永久保留缺陷。决定改进后更新对应契约与断言，保留能复现该场景的数据。

## 7. 本次验证

2026-09-10 全量实际结果：**59 passed，0 failed，0 errors，0 skipped**。
报告：reports/raw/daily-direct-20260910.xml。

这些结果证明给定数据上的软件行为，不证明真实模型质量、PDF 转换、完整 Gradio 页面、DeepEval 分数或主项目独立实现完成。默认测试范围仍为 daily、unit、integration；历史 tests/upstream 不进入默认收集，避免旧替身影响新版实验。

每次学习记录：日期/学习日、输入、调用函数、预期、实际通过或失败、解释、自己新增的反例、尚未验证的在线/人工部分。
