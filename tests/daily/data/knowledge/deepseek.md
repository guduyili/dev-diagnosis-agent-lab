# response_format 错误记录

用户报告的实际错误文本是：This response_format type is unavailable now。
本地 rewrite_query 使用 with_structured_output(QueryAnalysis, method="function_calling")。
它将结构化结果放进工具参数，避免依赖 json_schema 响应格式。
离线构造成功不能证明 DeepSeek 在线请求已经成功。
