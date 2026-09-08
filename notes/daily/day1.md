# QueryAnalysis

查询改写节点的协议：清晰度，可检索问题和澄清提示

- is_clear,questions,clarification





## config.py

模型、分块、top-k、循环上限

- chunk: 先按照 Markdown 标题形成parent， 再把parent 切成适合向量召回的child，parent 尺寸过小会缺上下文，过大会增加生成成本；应通过实验而不是凭感觉调整。



## document_chunker

把 Markdown 文档转换成 parent/child 两级 Document。

父块保留较完整的章节的上下文，子块较短，适合向量召回

流程先按标题切分，再合并过小章节，拆分过大章节，最后修补边界过小的块，因此这里的长度是字符数近似，不是token数。每一步都必须保留metadata，才能从child回到源文件和parent





