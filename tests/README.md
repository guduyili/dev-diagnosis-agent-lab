# 测试入口

tests/daily 已按学习日程重写，直接导入 references/upstream/agentic-rag-for-dummies/project 的真实模块。
59 项离线检查覆盖 18 个学习时段；6 个时段另需人工证据。用法、来源、测试替身与改进方向见 [每日指南](../docs/11-daily-tests.md)。

~~~powershell
Set-Location 'G:\Learning\dev-diagnosis-agent-lab'
uv sync --frozen
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest tests/daily --day D06 -v
.\.venv\Scripts\python.exe -m pytest tests/daily --through-day D12 -q
.\scripts\check_day.ps1 -Day D06
~~~

- daily：测试真实上游工作树，使用真实 LangGraph、本地 Qdrant、parent JSON；只在模型边界使用替身。
- unit/integration：留给学习者独立实现的 src/diagnosis_agent 普通测试。
- upstream：早期的 13 项烟雾测试，含运行时替身，作为历史材料保留；已退出默认收集，优先使用新版 daily，勿在同一进程混跑。
- evals：今后显式运行真实模型质量评测，当前 daily 不调用 DeepEval。

旧版 daily 与指南已备份到 notes/archive/daily-before-upstream-20260909/，不参与 pytest 收集。
日程通过不代表独立实现或真实模型效果通过；主项目 src 中的练习代码继续由学习者完善。
