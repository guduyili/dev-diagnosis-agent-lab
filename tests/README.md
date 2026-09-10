# 测试入口

tests/daily 按“一个自动学习日一个 test_dXX 文件”组织。18 个文件、59 项测试；另外 6 个学习时段用人工证据验收。

当前文件内直接写上游 import、数据、对象创建、调用和断言。模型响应和 embedding 替身也完整放在当天文件，不再通过 support.py 或业务 fixture 隐藏执行过程。

~~~powershell
Set-Location 'G:\Learning\dev-diagnosis-agent-lab'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest tests/daily/test_d05_chunking.py -v
.\.venv\Scripts\python.exe -m pytest tests/daily --day D05 -v
.\scripts\check_day.ps1 -Day D12 -Cumulative
~~~

新环境先执行 uv sync --frozen。详细日程、可修改方向和测试辅助工具解释见 [每日指南](../docs/11-daily-tests.md)。

- daily：直接调用本地 agentic-rag-for-dummies/project，实际执行 LangGraph、本地 Qdrant 与 parent JSON。
- unit/integration：留给自己实现的 src/diagnosis_agent。
- upstream：早期烟雾测试作为历史材料保留，已退出默认收集；不要与 daily 同进程混跑。
- evals：今后显式运行真实模型质量评测；daily 不调用在线裁判。

前一版 weekly 文件、support 与数据已备份至 notes/archive/daily-fixtures-20260910/。业务源码由学习者继续完善。
