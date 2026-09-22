from pathlib import Path
from langgraph.graph import StateGraph, START, END

from agents.ingest.schemas import IngestGraphState, IngestGraphContext
from agents.ingest.nodes.pdf_to_md import pdf_to_md
from agents.ingest.nodes.md_img_handle import md_img_handle
from agents.ingest.nodes.md_split import md_split
from agents.ingest.nodes.entity_recognize import entity_recognize
from agents.ingest.nodes.text_to_embedding import text_to_embedding
from repositories.milvus_repository import MilvusEntityRepository, MilvusChunkRepository
from clients.milvus import milvus_client

graph_builder = StateGraph(state_schema=IngestGraphState, context_schema=IngestGraphContext)

graph_builder.add_node(pdf_to_md)
graph_builder.add_node(md_img_handle)
graph_builder.add_node(md_split)
graph_builder.add_node(entity_recognize)
graph_builder.add_node(text_to_embedding)

graph_builder.add_edge(START, "pdf_to_md")
graph_builder.add_conditional_edges("pdf_to_md", lambda state: "md_img_handle" if state.should_continue else END)
graph_builder.add_conditional_edges("md_img_handle", lambda state: "md_split" if state.should_continue else END)
graph_builder.add_conditional_edges("md_split", lambda state: "entity_recognize" if state.should_continue else END)
graph_builder.add_conditional_edges("entity_recognize",
                                    lambda state: "text_to_embedding" if state.should_continue else END)
graph_builder.add_edge("text_to_embedding", END)

graph = graph_builder.compile()

if __name__ == '__main__':
    import asyncio


    async def main():
        # 项目根目录：本文件位于 agents/ingest/ 下，向上两级即项目根
        project_root = Path(__file__).resolve().parents[2]

        file_path = project_root / "test" / "test_file" / "HAK180.pdf"
        markdown_dir = project_root / "test" / "test_md_dir"
        state = IngestGraphState(
            file_path=file_path,
            markdown_dir=markdown_dir
        )
        milvus_entity_repository = MilvusEntityRepository(milvus_client.client)
        milvus_chunk_repository = MilvusChunkRepository(milvus_client.client)

        await milvus_entity_repository.ensure_collection()
        await milvus_chunk_repository.ensure_collection()
        context = IngestGraphContext(
            milvus_entity_repository=milvus_entity_repository,
            milvus_chunk_repository=milvus_chunk_repository
        )

        async for chunk in graph.astream(state, context=context, stream_mode="custom"):
            print(chunk)


    asyncio.run(main())
