# 用例格式与维护

每行一个 JSON 对象，UTF-8。`dev.jsonl`、`regression.jsonl`、`holdout.jsonl` 当前均为空。
`../examples/case-format.jsonl` 只演示格式，不可自动加入正式评测。

必填字段：
- `id`：唯一用例 ID。
- `family_id`：同一个 bug/相似改写所属家族，用于避免跨数据集泄漏。
- `origin`：real / constructed / synthetic_example。
- `review_status`：pending / approved；只有 approved 才能进入正式运行。
- `category`：knowledge / clarification / tool / recovery 等。
- `input`：仅包含 Agent 可见的信息。
- `expected`：人工参考状态、关键事实、来源 ID、允许的工具行为，不能传给 Agent。
- `source_refs`：来源文件或可验证记录的相对路径列表。

后续可添加 reviewer、reviewed_at、created_at、project_version。时间必须真实。

评分程序应拒绝空用例集、重复 ID、未审核用例、无依据的 expected；加载数据时检查近重复和 family_id 是否跨集合。
变更用例必须解释原因并保留版本，不能为提高分数悄悄修改标准答案。
