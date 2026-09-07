# 上游与来源登记

业务参考：https://github.com/GiovanniPasq/agentic-rag-for-dummies
评测依赖：https://github.com/confident-ai/deepeval

上游已下载到 `upstream/agentic-rag-for-dummies/`，当前基线 SHA 为 `2461e5251c6b9a6be71d13176ab43301f3c0a068`。
本地工作树在该基线之上将聊天模型切换为 DeepSeek V4 Flash，并补充了 OpenAI 兼容 API 配置；这是本学习项目的个人改动，不是上游原始实现。
上游当前说明分别标注 MIT 与 Apache-2.0；引入时重新检查具体文件和依赖的许可。

第 1 周可把上游克隆到本目录下的 `upstream/`，该目录已被忽略。学习基线和自己的环境分开。

| 项目 | 本地位置 | commit SHA / 版本 | 核查日期 | 运行结果 |
|---|---|---|---|---|
| agentic-rag-for-dummies | `upstream/agentic-rag-for-dummies/` | `2461e5251c6b9a6be71d13176ab43301f3c0a068` + 本地未提交修改 | 2026-09-07 | 已完成静态检查；未发起模型请求 |
| deepeval | 待安装 | 待记录 | 待记录 | 未运行 |

复制源码时在个人提交说明中记录原路径与 SHA，并保留对应版权与许可。不要把本练习项目描述成全部原创。
