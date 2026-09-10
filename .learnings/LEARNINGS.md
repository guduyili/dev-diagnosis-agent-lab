# 可复用经验

记录经验证的纠正、知识缺口和方法改进。具体格式可参考问题和实验模板。
每条记录至少包含：日期、来源、问题、正确做法、验证证据、适用范围、状态。

## [LRN-20260907-001] best_practice

**Priority**: medium
**Status**: promoted
**Area**: docs

### Summary
区分空项目、虚构示例与真实运行结果，才能持续建立可信的学习成果。

### Details
本项目初始只有骨架与文档；正式用例集为空，格式示例标记 synthetic_example/pending。
所有业务运行和评测结果必须在实际执行后登记。

### Metadata
- Source: conversation
- Related Files: README.md, docs/09-progress.md, data/cases/README.md
- Promoted: AGENTS.md

## [LRN-20260909-CMT] correction

**Logged**: 2026-09-09
**Priority**: medium
**Status**: resolved
**Area**: docs

### Summary
Agent 学习注释要区分提示词要求、图的硬约束、实际工具结果和 UI 展示。

### Details
本次修正了首步检索被强制执行、检索 key 代表成功、共享 Gradio 实例自动隔离用户、文件导入完全回滚等过度表述。补充 reducer 合并示例、计数边界、父子块 ID、结构化输出及评测上下文的实际来源。函数装饰为工具后，其 docstring 可能成为模型输入，不能当成普通学习注释任意扩写。

### Validation
使用修改前快照比较剔除普通文档字符串后的可执行 AST，并额外逐字比较工具和 QueryAnalysis 的模型可见 docstring；不通过导入应用初始化模型来验证纯注释修改。

### Metadata
- Source: conversation
- Related Files: `references/upstream/agentic-rag-for-dummies/project/rag_agent/nodes.py`, `references/upstream/agentic-rag-for-dummies/project/rag_agent/tools.py`

## [LRN-20260909-DAILY] correction

**Logged**: 2026-09-09
**Priority**: high
**Status**: resolved
**Area**: tests

### Summary
用户要求学习检验直接调用已有 Agent 项目；不能将尚不存在的规划接口作为当前必过契约。

### Details
重写 daily 后，真实导入上游模块并执行实际 LangGraph/Qdrant；仅在外部模型边界提供响应或向量。将未实现的评测、报告解析功能列为扩展。旧测试备份，学习者源码保留。使用测试夹具需注明来源与替换范围，不能把固定响应检验写成真实模型质量。

### Validation
59 项真实上游离线行为检查通过；人工日单列。详见 docs/11-daily-tests.md。

## [LRN-20260910-DAILY] correction

**Logged**: 2026-09-10
**Priority**: high
**Status**: resolved
**Area**: tests

### Summary
初学者要能在当天文件看到完整实验；真实调用上游代码仍不足以保证学习可读性。

### Details
用户指出多层 fixture、support 和外部数据让理解测试需要跨文件追踪。按天展开准备数据、构造真实对象、调用和断言，仅保留少量公共路径/离线设置。模型边界用本地可见的 Mock，固定 embedding 类在当天文件完整列出，接受适量重复代码。

### Validation
保留 59 项检查并实际通过；旧版完整备份。学习用代码组织优先考虑可观察的调用过程，不应优先消除所有重复。
