# thread_id 与检查点

create_agent_graph 使用 InMemorySaver 保存进程内状态。
同一个 thread_id 用于查询、更新和恢复同一检查点。
不同 thread_id 可保存不同的图状态。
Gradio 当前共享一个 RAGSystem，不能因此声称页面已自动隔离不同用户。
