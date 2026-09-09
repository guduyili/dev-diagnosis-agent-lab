"""文件转换、目录清理和上下文长度估算工具。"""

import os
import shutil
import config
# 这是模块级导入：即使调用者只需要目录清理，导入 utils 也需要 PDF 依赖可用。
import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import glob
import tiktoken
from functools import lru_cache


def clear_directory_contents(directory: Path) -> None:
    """删除目录内的全部内容，保留目录本身；传入非目录时直接返回。"""
    # 子目录通过 rmtree 递归删除，不是仅删除第一层文件；本函数没有允许目录
    # 白名单或路径边界校验，调用者必须明确传入要清空的数据目录。
    directory = Path(directory)
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 关闭 tokenizer 内部并行机制，不只是隐藏一条提示；赋值在导入模块时生效。

def pdf_to_markdown(pdf_path, output_dir):
    """将单个 PDF 转成 Markdown，并丢弃图片以保持文本检索路径简单。"""
    # ignore_images/write_images 控制图片处理，并不承诺扫描件一定能抽出文字，
    # 也不保证表格结构完整；评估召回前应先检查转换后的 Markdown。
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
    # 允许先编码代理字符，再忽略无效 UTF-8 序列，避免写盘编码报错；
    # 该处理可能丢弃异常字符，并非无损文本修复。
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    """批量转换匹配路径的 PDF；默认不覆盖已有 Markdown。"""
    # path_pattern 既可为单一路径，也可含 glob 通配符；无匹配时循环不执行。
    # 此方法没有额外按后缀过滤，也没有排序，调用方应传入合适的 PDF 路径模式。
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)

@lru_cache(maxsize=1)
def _get_token_encoding():
    """懒加载 tokenizer，并在模型映射不可用时回退到 cl100k_base。"""
    # 无参函数只缓存一个结果，避免每次估算都初始化；失败返回的 None 也会缓存。
    # 若环境修好后希望重试，需要清除此函数缓存或重启进程。
    try:
        return tiktoken.encoding_for_model("gpt-4")
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def estimate_context_tokens(messages: list) -> int:
    """估算消息内容的 token 数，用于上下文压缩阈值而非精确计费。"""
    # 只计算非空 content，忽略 tool_calls 参数、角色和协议开销；多模态内容
    # 只是 str 转换，不是真正多模态 token 计算。GPT-4 分词也不等于 DeepSeek 分词。
    # 无编码器时按字符数 //4 粗估，中文等文本误差可能很大，不可当成服务端用量。
    contents = [
        str(msg.content)
        for msg in messages
        if hasattr(msg, "content") and msg.content
    ]
    encoding = _get_token_encoding()
    if encoding is None:
        return sum(max(1, len(content) // 4) for content in contents)
    return sum(len(encoding.encode(content)) for content in contents)
