import json
import re
from langgraph.runtime import Runtime
from langchain_text_splitters import MarkdownHeaderTextSplitter

from core.log import logger
from vendors.markdown_chunker.chunking_strategy import MarkdownChunkingStrategy
from agents.ingest.schemas import IngestGraphState, IngestGraphStepInfo, IngestMarkdownChunk, IngestGraphContext
from agents.ainvoke_llm import ainvoke_llm_str

# 匹配 Markdown 表格的表头行 + 分隔符行模式
_PATTERN_MD_TABLE = re.compile(r"^\|.+\|$")
_PATTERN_MD_TABLE_SEP = re.compile(r"^\|(\s*[-:]+\s*\|)+$")
# 匹配 HTML 表格标签（<table>, <tr>, <td>, <th>）
_PATTERN_HTML_TABLE = re.compile(r"<(/?(?:table|tr|t[dh])\b[^>]*>)", re.IGNORECASE)


def _has_table(content: str) -> bool:
    """检查内容中是否包含表格（Markdown 表格或 HTML 表格标签）"""
    # 1. 检测 Markdown 表格：表头行 + 分隔符行模式
    lines = content.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _PATTERN_MD_TABLE.match(stripped):
            if i + 1 < len(lines) and _PATTERN_MD_TABLE_SEP.match(lines[i + 1].strip()):
                return True

    # 2. 检测 HTML 表格标签：<table>, <tr>, <td>, <th>
    if _PATTERN_HTML_TABLE.search(content):
        return True

    return False


async def _table_to_natural_language(content: str) -> str:
    """调用 LLM 将内容中的表格（Markdown 或 HTML）转换为自然语言描述，替换原表格"""
    try:
        result = await ainvoke_llm_str("table_to_text", {"content": content})
        return result
    except Exception as e:
        logger.warning(f"表格转自然语言失败，保留原始内容: {e}")
        return content


async def md_split(state: IngestGraphState, runtime: Runtime[IngestGraphContext]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="md文档分割", status="running"))

    try:
        # 1.按标题切分
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6")
            ],
            strip_headers=True
        )
        header_chunks = header_splitter.split_text(state.markdown_content)

        # 2.每个标题下的内容进行切分(需要保留：标题，内容块，所属标题的内容块编号)
        strategy = MarkdownChunkingStrategy()
        markdown_chunks: list[IngestMarkdownChunk] = []
        for header_chunk in header_chunks:
            # 处理标题
            if not header_chunk.metadata:
                # 没有标题内容块的需要用llm生成一个标题
                result = await ainvoke_llm_str("generate_chunk_title", {"content": header_chunk.page_content})
                chunk_title = {"h1": result}
            else:
                chunk_title = header_chunk.metadata
            title = json.dumps(chunk_title, ensure_ascii=False)

            # 处理内容
            raw_content = header_chunk.page_content
            content = raw_content.replace("\r\n", "\n").replace("\r", "\n")

            # 讲表格转换为自然语言
            if _has_table(content):
                content = await _table_to_natural_language(content)

            chunks = strategy.chunk_markdown(content)
            for index, chunk in enumerate(chunks):
                markdown_chunk = IngestMarkdownChunk(
                    file_name=state.file_path.name,
                    file_url=state.file_url,
                    title=title,
                    content=chunk,
                    header_chunk_index=index
                )
                markdown_chunks.append(markdown_chunk)
    except Exception as e:
        writer(IngestGraphStepInfo(name="md文档分割", status="failed", error=str(e)))
        return {"should_continue": False, "error": str(e)}

    writer(IngestGraphStepInfo(name="md文档分割", status="success"))
    # logger.info(markdown_chunks)
    return {"markdown_chunks": markdown_chunks}


