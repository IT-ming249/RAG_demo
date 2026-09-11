from langgraph.runtime import Runtime
from langchain_text_splitters import MarkdownHeaderTextSplitter

from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo


async def md_split(state: IngestGraphState, runtime: Runtime[IngestGraphStepInfo]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="md文档分割", status="running"))

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

    # 2.每个标题下的内容进行切分
