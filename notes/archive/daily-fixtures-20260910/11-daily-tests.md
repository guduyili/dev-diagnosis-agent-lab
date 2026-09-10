# 每日检验：直接测试 agentic-rag-for-dummies

更新：2026-09-09。按本次要求重写 tests/daily，测试对象是本地上游参考实现：

G:\Learning\dev-diagnosis-agent-lab\references\upstream\agentic-rag-for-dummies\project

当前有 **59 项离线自动测试，覆盖 18 个学习时段**；另 6 个时段需要人工证据。D01～D24 沿用八周手册的 A/B/C 编号，每个时段可以分多天完成。这些测试帮助你理解和修改上游实现，通过不代表你已完成自己的诊断 Agent。

原先指向尚未实现的 diagnosis_agent、evals 接口的 daily 已被替换，旧文件保存在 notes/archive/daily-before-upstream-20260909/，以 .py.txt 结尾，不参与收集。主项目 src 中的学习代码没有被补齐或替换。

## 1. 每天怎么运行

在主项目根目录使用主项目 Python，测试会检查导入的业务模块确实来自上游 project：

~~~powershell
Set-Location 'G:\Learning\dev-diagnosis-agent-lab'
uv sync --frozen
$env:PYTHONUTF8 = '1'

# 查看当天有哪些测试，不执行
.\.venv\Scripts\python.exe -m pytest tests/daily --day D06 --collect-only -q

# 只跑当天；W02-C 与 D06 等价
.\.venv\Scripts\python.exe -m pytest tests/daily --day D06 -v

# 找到失败就停止；需要查看局部变量时可追加 --pdb
.\.venv\Scripts\python.exe -m pytest tests/daily --day D06 -x -v

# 累计复习
.\.venv\Scripts\python.exe -m pytest tests/daily --through-day D12 -q

# 单条检验
.\.venv\Scripts\python.exe -m pytest tests/daily/test_w04_agent.py -k clarification -v

# 全部自动检查
.\.venv\Scripts\python.exe -m pytest tests/daily -q
~~~

快捷入口仍为：

~~~powershell
.\scripts\check_day.ps1 -Day D06
.\scripts\check_day.ps1 -Day W04-C -Cumulative
~~~

脚本将 JUnit XML 写入 reports/raw/daily-upstream-日期标识.xml，并保留 pytest 退出码。报告是实际 pytest 生成的数据，当前上游没有读取 JUnit 的业务工具。

默认执行 pytest 会收集 daily、unit、integration；早期 tests/upstream 是独立保留的旧烟雾测试，不再放进默认收集。请优先使用新版 daily，旧烟雾测试含运行时替身，不能代表真实图验收，也不要与 daily 混在一个进程运行。

## 2. 哪些是真的，哪些被替换

| 层 | 执行内容 | 能证明什么 |
|---|---|---|
| 上游业务代码 | 正常 import DocumentChunker、DocumentManager、ParentStoreManager、VectorDbManager、节点、工具、日志等 | 实际函数在给定数据上的行为 |
| LangGraph | 真实 StateGraph、reducer、Send、ToolNode、InMemorySaver | 状态合并、真实路由、工具执行、暂停恢复 |
| 存储和检索 | 真实临时 JSON、QdrantClient、QdrantVectorStore、HYBRID 查询 | 写入、持久化、检索与 parent 关联 |
| embedding | FixedDenseEmbeddings、FixedSparseEmbeddings，仅替换模型构造器 | 固定向量下的检索行为；不代表 Qwen/BM25 语义能力 |
| 节点的模型响应 | ScriptedLLM 记录输入，按预设队列返回结果 | 调用顺序、次数、证据如何进入提示词；不代表回答质量 |
| D12 HTTP 协议 | 真实 ChatOpenAI/OpenAI SDK + httpx.MockTransport | 请求体包含 QueryAnalysis 工具定义，未使用 json_schema 响应格式 |
| token 估算 | 执行上游已有 len//4 回退分支 | 该分支的行为；不代表服务端准确 token 计费 |
| UI/追踪 | 调用上游消息转换函数、禁用状态的 Observability | 数据转换和关闭追踪的行为；不启动 Gradio 页面或 Langfuse 服务 |

不替换被测业务模块、不复制路由算法、不伪造 MessagesState/Send。模型响应、故障注入是测试输入；真正执行分块、构建提示词、路由、检索和提取证据的仍是上游代码。

SDK 协议测试由 MockTransport 接收实际序列化的请求，因此没有访问 DeepSeek。真实 key、在线兼容性、模型质量、PDF 转换和完整浏览器交互需要另做实验。本轮不安装/调用 DeepEval 裁判；上游 project 本身没有本手册设想的 DeepEval adapter、JUnit 解析器或回归统计模块。

## 3. 按原学习日程的对应表

“拓展”是你以后可以选择实现的方向，不是当前已实现能力，也不要求今天全部完成。

| 学习日 | 原手册主题 | 本次真实调用与检查 | 适合自行拓展 |
|---|---|---|---|
| D01 / W01-A | 环境与配置 | config 来源、DocumentChunker 参数校验、QueryAnalysis、RAGSystem.initialize/get_config/reset_thread | 修改默认配置后观察影响；新增参数校验 |
| D02 / W01-B | 请求追踪 | create_agent_graph 的真实节点顺序、返回状态与模型调用次数 | 自行画检索/澄清/无证据三条流程；另做在线记录 |
| D03 / W01-C | 真实问题记录 | 人工：400 的触发条件、来源和验证层级 | 增加一个有来源的实际 incident |
| D04 / W02-A | 资料与来源 | DocumentManager.add_documents/get_markdown_files，真实入库、重复文件与空文件 | 内容修订检测、同名路径处理 |
| D05 / W02-B | 分块 | create_chunks_single/create_chunks，标题、代码、编号片段和父子关联 | 换成自己项目的一份说明文档 |
| D06 / W02-C | 边界与反例 | 缺文件、空目录、非法配置、短文和默认 .pdf 来源标签 | 修正默认来源标签或完善超长代码策略 |
| D07 / W03-A | 存储与索引 | parent 写回/覆盖/排序，真实 Qdrant 关闭重开、维度不兼容 | 加模型版本标识、索引清单 |
| D08 / W03-B | 检索与回取 | ToolFactory.create_tools 后真实 invoke；查询/k/阈值传递、按 ID 回取 | 新增有人工来源标注的检索问题 |
| D09 / W03-C | 生成与实验 | orchestrator 的工具上下文、aggregate_answers 排序、fallback 空证据输入 | 用真实模型比较答案；上游没有独立 baseline.diagnose |
| D10 / W04-A | 状态与路由 | 真实 reducer 的累加/去重/reset，真实 Send 的问题与序号 | 测更多并行子问题、消息替换规则 |
| D11 / W04-B | 工具与预算 | 预算边界、真实 ToolNode、schema 参数错误、工具错误哨兵 | 独立记录 proposed/executed/succeeded |
| D12 / W04-C | 澄清与恢复 | 真检查点中断、显式双 thread 隔离、恢复后 pending 清空、真实 SDK 序列化 | 增加多次澄清；随后改造 UI 的用户隔离 |
| D13 / W05-A | 评测资料 | _retrieval_contexts 与 collect_answer 使用实际工具输出 | 再将结果适配为 DeepEval 输入；不得注入参考答案 |
| D14 / W05-B | 上下文与记录 | 真压缩节点、RemoveMessage、证据保留、历史摘要、压缩阈值 | 比较召回文本与最终模型收到的上下文 |
| D15 / W05-C | 裁判校准 | 人工：接入裁判后的评分对照 | 正确/错误/部分正确/应澄清样例 |
| D16 / W06-A | 研发可观测性 | 当前已有 logged_node 的真实返回、异常与日志，token 估算边界 | 原计划 JUnit 解析器为未来个人实现 |
| D17 / W06-B | 工具结果展示 | ChatInterface 工具 ID 关联、显示截断、JSON 缓冲、禁用追踪 | 上游没有报告读取工具；可之后新增 |
| D18 / W06-C | 实际使用 | 人工：真实任务、修正量和时间记录 | 至少记录一次失败与手动修正 |
| D19 / W07-A | 回归链路 | 完整图的搜索→parent→子答案→聚合，第二轮不混入旧子答案 | 增加你的真实缺陷回归 |
| D20 / W07-B | 失败门禁 | 超预算整批未执行、真实图传播模型错误 | 后续将 pytest 退出码接入 CI |
| D21 / W07-C | 红→绿 | 人工：改真实源码，观察失败，再恢复 | 不通过降低断言强度使报告变绿 |
| D22 / W08-A | 单因素对照 | 同一真实临时索引上 k=1/k=3 的结果比较 | 固定其他因素后评估真实 embedding 与标注集 |
| D23 / W08-B | 展示 | 人工：成功、失败、回归各一个 | 说明本地库与模型替身各自证明什么 |
| D24 / W08-C | 独立讲解 | 人工：贡献、许可、可复现命令 | 用新输入独立解释关键模块 |

D16/D17 用上游现有日志与 UI 转换作为前置学习检验，没有把原计划中的 JUnit 功能说成已经存在。它们通过也不代表原手册的报告工具开发任务完成。

## 4. 代码入口

| 文件（相对于 tests/daily） | 学习日 |
|---|---|
| test_w01_start.py | D01–D02 |
| test_w02_knowledge.py | D04–D06 |
| test_w03_rag.py | D07–D09 |
| test_w04_agent.py | D10–D12 |
| test_w05_evaluation.py | D13–D14 |
| test_w06_observability.py | D16–D17 |
| test_w07_w08_regression.py | D19、D20、D22 |
| conftest.py | 来源检查、按日选择、离线保护、真实临时库 |
| support.py | embedding/模型响应边界；不含业务算法 |
| data/ | 有来源说明的 Markdown 与检索期望 |

D08 的数据测试示例：测试先把 data/knowledge 中的文件交给真实 DocumentManager 入库，再通过真实工具调用取得结果：

~~~python
tools = {tool.name: tool for tool in tool_factory.create_tools()}
result = tools["search_child_chunks"].invoke({"query": case["query"], "limit": 1})
assert f"File Name: {case['expected_source']}" in result

parent_id = result.split("Parent ID: ", 1)[1].splitlines()[0]
parent = tools["retrieve_parent_chunks"].invoke({"parent_id": parent_id})
assert case["expected_text"] in parent
~~~

这里的 parent_id 从实际检索结果解析，不是用固定字符串冒充召回。D19 的完整链路则使用已经由真实导入生成的 deepseek_p0，预设模型会请求这个 ID，再验证该工具结果是否真的进入下一次模型输入。

## 5. 按学习进度使用和改进

1. 先读当天测试的输入、被调用函数和断言，写下预期结果。
2. 运行单条测试，沿 traceback 或调试器查看上游函数的实际中间值。
3. 换一份资料或一个边界输入，保持本次只改变一个因素。
4. 若修改上游源码，重跑单条、当天和累计测试。tests/daily 导入的是工作树，修改会在新 pytest 进程生效。
5. 如要验证自己独立实现的模块，另建 tests/unit 的对应测试并显式改 import；不要让导入失败时自动回退到上游。
6. 日记区分“自动测试通过”“能独立解释”“在线实验成功”三项证据。

带 current_behavior 标记的测试记录已知现状：

- 图没有强制首步检索；
- 同名 Markdown 会跳过更新；
- 默认 source_name 可能给 Markdown 标为 .pdf；
- 失败的 parent 请求仍进入 retrieval_keys。

它们不是要求你永久保留缺陷。如果决定改进，先写新契约和会失败的目标测试，实现后更新对应现状断言，同时保留故障场景。不要简单删掉测试。

给数据集新增文件时，更新 data/README.md 的来源、retrieval_cases.json 的标注，以及 D04 的期望文件列表。给 ScriptedLLM 新增响应时，说明它对应哪一次模型调用；assert_consumed 会发现多调用、少调用或路径变化。

## 6. 结果与验证边界

2026-09-09 实际运行：59 passed，未使用 skip/xfail 隐藏失败。完整命令：

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/daily -q --junitxml=reports/raw/daily-upstream-20260909.xml
~~~

报告位置：reports/raw/daily-upstream-20260909.xml。该报告是软件行为检验，不是 RAG 质量评分、模型可用性证明或主项目实现完成率。

选择器与脚本实测：D06 为 6 passed、53 deselected；check_day.ps1 -Day W04-C -Cumulative 为 39 passed、20 deselected，并列出 D03 人工待验收。默认 pytest 收集结果也为 59 passed。

- passed：当前选择的测试通过。
- deselected：未选择的学习日，没有运行。
- FAILED/ERROR：断言或运行错误；缺依赖也会显式报错，不安装假模块遮盖。
- MANUAL_NOT_CHECKED：仍需人工记录，不计入自动通过数。
- 单独选择人工日会给出说明并返回退出码 4。
- 退出码 0/1/4/5 分别表示通过、有失败、参数/人工日提示、未收集到测试。

默认禁止本进程常用 Python TCP 连接，知识库写入 pytest 临时目录。缺少真实依赖时执行 uv sync --frozen；不要通过修改 sys.modules 来补一个假的 LangGraph。

实际依赖版本由主项目 uv.lock 固定；本次安装的是普通检验所需子集，没有安装整套上游 Notebook、Gradio 和在线评测环境。

## 7. 每日记录模板

~~~text
学习日：
执行命令：
被测上游文件/函数：
输入资料及来源：
预期结果：
实际通过/失败/错误/未运行数：
我能解释的状态变化：
发现的缺陷或限制：
自己新增的反例：
本次是否使用模型/embedding 替身：
在线或人工任务尚缺什么：
下一步：
~~~
