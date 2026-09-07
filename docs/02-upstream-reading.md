# 上游源码阅读与迁移

## 两个仓库的角色

- `agentic-rag-for-dummies`：业务实现参考。先完整观察一次，再把需要的设计迁入本项目。
- `deepeval`：使用其库与文档建立评测，不需要先修改框架内部。

本项目尚未克隆上游、Fork 远程仓库或复制源码。

## 第 1 周：下载参考版本

在本项目根目录执行（若目标已有内容，先检查，勿重复克隆）：

```powershell
git clone https://github.com/GiovanniPasq/agentic-rag-for-dummies.git references/upstream/agentic-rag-for-dummies
git -C references/upstream/agentic-rag-for-dummies rev-parse HEAD
```

把实际 SHA 填入 [来源登记](../references/README.md)。如果你希望未来向上游贡献，先在 GitHub 页面 Fork，然后克隆你自己的 Fork，并添加原仓库为 upstream。不要把学习骨架误当成已有 Fork。

按已选 commit 的说明建立参考环境。验证顺序：Notebook → 单次检索 → `project/app.py`。跑通结果包括实际截图、输入、输出与异常记录，不预设一定跑通。

## 阅读顺序与问题

以下路径属于上游仓库，在本项目尚不存在；以选定 SHA 的目录为准。

| 顺序 | 上游路径 | 必须回答的问题 | 留下的笔记 |
|---|---|---|---|
| 1 | README、notebooks/ | 普通 RAG 与循环检索各在哪一步？ | 一张流程草图 |
| 2 | project/config.py | 模型、分块、top-k、循环上限在哪里？ | 参数表 |
| 3 | project/core/document_manager.py、project/document_chunker.py | 如何从资料得到 child/parent？如何保留来源？ | 一份分块样例 |
| 4 | project/db/vector_db_manager.py、parent_store_manager.py | 子块如何找回父块？如何处理重复文档？ | 检索返回结构 |
| 5 | project/rag_agent/graph_state.py、graph.py | 状态如何更新？哪些内容跨轮保留？ | 状态字段表 |
| 6 | project/rag_agent/nodes.py、edges.py | 何时澄清、重写、检索、结束？ | 三种分支轨迹 |
| 7 | project/rag_agent/tools.py、schemas.py | 工具输入如何验证？错误如何回传？ | 工具契约 |
| 8 | project/core/chat_interface.py、observability.py | 流式结果和执行记录如何区分？ | 一次完整 trace |

先只弄懂“一条问题”的流动；上游已有并行子问题等能力，不需要第一周就复现全部功能。

## 迁移顺序

1. 先写本项目的数据结构和测试，再引入所需检索逻辑。
2. 先得到不用复杂图的普通 RAG 基线。
3. 再添加 query 澄清、检索决策和有界循环。
4. 将工具调用和证据结构暴露给评测层，不让 DeepEval 依赖 Gradio UI。

每次移植记录：原文件、SHA、修改原因、个人新增逻辑、保留的许可说明。对比上游已有功能时不要声称自己原创实现了所有模块。

## 维护上游

每月查看一次上游变更，先在参考目录试验更新，再决定是否迁移。不要在关键评测前顺手更新所有依赖。

出处：[上游模块说明](https://github.com/GiovanniPasq/agentic-rag-for-dummies/tree/main/project)。阅读问题和迁移顺序为本学习项目的设计建议。
