# Windows 环境与依赖管理

## 创建时的机器检查

2026-09-07 已发现 Git 和 uv；Python launcher 列出了 3.11、3.12、3.13 等解释器，但 3.11 的登记路径无法启动。已验证可运行的 Python 3.12 路径为 `D:\python 3.12.4\python.exe`，默认 `python` 指向可运行的 3.13。本路线先用 3.12 创建独立环境；这是环境选择，不代表已验证所有业务依赖组合。

## 1. 创建本项目环境（以下命令需要你执行）

在 PowerShell 中：

```powershell
Set-Location 'G:\Learning\dev-diagnosis-agent-lab'
& 'D:\python 3.12.4\python.exe' -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

最后一条只安装当前空包，不会安装 LangGraph、DeepEval 或模型。直接调用虚拟环境解释器即可，不必修改 PowerShell 执行策略。

创建 venv 前先检查是否已有 `.venv`；有则复用并核对版本，不重复重建。

## 2. 按阶段添加依赖

- 第 2 周编写普通测试时：安装 `pytest`。
- 第 3 周实现检索时：根据已选上游版本引入实际使用的 LangGraph、检索与模型适配依赖。
- 第 5 周：在本项目环境安装 `deepeval`，先验证一条用例。

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
# 到第 5 周才执行下一条
.\.venv\Scripts\python.exe -m pip install deepeval
.\.venv\Scripts\python.exe -m pip check
```

这些是首次探索安装命令，不是已验证的版本组合。跑通后把直接依赖与已验证版本写入 `pyproject.toml`。使用 uv 时再生成、提交 `uv.lock`；以后复现使用 `uv sync --frozen`。不要把上游 requirements 与最新 DeepEval 直接混装而不检查兼容性。

若你先用 pip 记录环境：

```powershell
.\.venv\Scripts\python.exe -m pip freeze --exclude-editable > references\environment-pip.txt
```

该文件是当前 Windows 环境快照，不自动等价于跨平台锁文件。切换依赖管理方案时，在独立环境验证后再采用。

## 3. 上游环境独立

`references/upstream/` 下的参考项目使用自己的 `.venv`，严格先按其选定 commit 的 README 和 requirements 安装。
上游应用目前默认走 Ollama；切换到云模型通常还需要改模型初始化代码和安装 provider 包，不能只填 `.env` 就认为完成适配。
即使聊天模型走云 API，检索的 embedding/sparse 模型仍可能需要下载和本地运行。

## 4. 模型方案记录

| 项目 | 必须记录 |
|---|---|
| 生成模型 | provider、模型 ID、接口地址、工具调用与结构化输出支持 |
| Embedding | 模型 ID、向量维数、下载位置、索引版本 |
| 裁判模型 | provider、模型 ID、评分规则、与生成模型是否相同 |
| 实验设置 | 温度、重试、超时、最大工具次数、数据集版本 |

`.env.example` 中的 `DIAG_*` / `EVAL_*` 是待实现的配置约定，不是已工作的集成；DeepEval 不会自动识别这些变量。先根据官方文档选择原生 provider 或自定义 adapter，并明确映射。

模型调用可能产生费用；先人工运行 1 条，再扩大到 5 条、完整开发集。不要默认启用付费定时回归。

## 5. 常见问题排查顺序

1. 确认虚拟环境解释器与安装依赖使用同一个路径。
2. 用 `pip check` 区分依赖冲突与模型错误。
3. 区分网络下载失败、模型认证失败、限流、embedding 维度不匹配。
4. 先验证单次模型请求和单次检索，再运行整个图。
5. 保存脱敏后的最小复现；不要通过全局降级包修复某个项目。

出处：[上游应用指南](https://github.com/GiovanniPasq/agentic-rag-for-dummies/tree/main/project)、[DeepEval 入门](https://deepeval.com/docs/getting-started)。
