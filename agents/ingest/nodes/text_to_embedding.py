from langgraph.runtime import Runtime

from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from integrations.embedding import generate_texts_embeddings
from dtos.milvus import MilvusInsertChunk


async def text_to_embedding(state: IngestGraphState, runtime: Runtime[IngestGraphStepInfo]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="文本转嵌入向量", status="running"))

    try:
        context = runtime.context

        # 1. 从markdown_chunks中提取文本内容
        assert state.markdown_chunks is not None
        texts = [f"标题：{chunk.title}, 内容：{chunk.content}" for chunk in state.markdown_chunks]

        # 2. 批量生成文本嵌入向量
        embeddings: list[tuple[list[float], dict[int, float]]] = []
        batch_size: int = 10
        for index in range(0, len(texts), batch_size):
            batch_texts = texts[index:index + batch_size]
            batch_embeddings = await generate_texts_embeddings(batch_texts)
            if batch_embeddings is not None:
                for vector in batch_embeddings:
                    embeddings.append((vector.get("dense"), vector.get("sparse")))

        # 3. 将生成的嵌入向量以及payload写入milvus
        milvus_chunk_repository = context.milvus_chunk_repository
        await milvus_chunk_repository.add_chunks([
            MilvusInsertChunk(
                file_name=chunk.file_name,
                title=chunk.title,
                content=chunk.content,
                header_chunk_index=chunk.header_chunk_index,
                entity_name=chunk.entity_name,  # 本块包含的实体（多个逗号拼接）
                dense_vector=embedding[0],  # 稠密向量
                sparse_vector=embedding[1]  # 稀疏向量
            )
            for chunk, embedding in zip(state.markdown_chunks, embeddings)
        ])
    except Exception as e:
        writer(IngestGraphStepInfo(name="文本转嵌入向量", status="failed", error=str(e)))

    writer(IngestGraphStepInfo(name="文本转嵌入向量", status="success"))