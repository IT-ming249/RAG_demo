import json

from langgraph.runtime import Runtime
from langchain_text_splitters import MarkdownHeaderTextSplitter

from core.log import logger
from vendors.markdown_chunker.chunking_strategy import MarkdownChunkingStrategy
from agents.ingest.schemas import IngestGraphState, IngestGraphStepInfo, IngestMarkdownChunk
from agents.ainvoke_llm import ainvoke_llm_str


async def md_split(state: IngestGraphState, runtime: Runtime[IngestGraphStepInfo]):
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

            chunks = strategy.chunk_markdown(content)
            for index, chunk in enumerate(chunks):
                markdown_chunk = IngestMarkdownChunk(
                    file_name=state.file_path.name,
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


