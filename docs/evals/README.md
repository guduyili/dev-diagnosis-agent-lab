# 回归评测实现预留

这里目前没有评分器、适配器或 DeepEval 测试。第 5 周开始实现。

推荐后续文件（尚未创建）：
- `case_loader.py`：加载用例，拒绝未审核用例和格式错误。
- `agent_adapter.py`：调用 Agent，返回真实答案、证据与工具记录。
- `test_regression.py`：用 `assert_test` 包装需要门禁的 DeepEval 指标。
- `run_experiment.py`：记录对比实验，导出数据与版本信息。

读取 [评测规范](../docs/05-evaluation.md)。评测脚本不得把标准答案当作 actual_output；执行失败不得自动填空答案再当作正常样本。
