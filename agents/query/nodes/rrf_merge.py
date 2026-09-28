from langgraph.runtime import Runtime

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from dtos.milvus import MilvusSearchChunk
from core.log import logger


async def rrf_merge(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(QueryGraphStepInfo(name="RRF排序", status="running"))

    embedding_chunks = state.embedding_chunks
    hyde_chunks = state.hyde_chunks

    try:
        assert embedding_chunks is not None
        assert hyde_chunks is not None

        # 1. 两路向量搜索结果按照得分倒序排序
        embedding_chunks.sort(key=lambda chunk: chunk.distance, reverse=True)
        hyde_chunks.sort(key=lambda chunk: chunk.distance, reverse=True)

        weights = (1.0, 1.0)
        k = 60  # 平滑常数, 通常为60
        rrf_chunks: dict[int, tuple[MilvusSearchChunk, float]] = {}

        # 2. 根据RRF公式计算向量搜索/假性文档搜索结果的RRF分数
        for index, embedding_chunk in enumerate(embedding_chunks, start=1):
            rrf_score = (1.0 / (k + index)) * weights[0]
            rrf_chunks[embedding_chunk.id] = (embedding_chunk, rrf_score)

        for index, hyde_chunk in enumerate(hyde_chunks, start=1):
            hyde_score = (1.0 / (k + index)) * weights[1]
            # 如果相关文档块在向量搜索结果中也出现了，需要叠加得分
            if hyde_chunk.id in rrf_chunks:
                existing_chunk, existing_score = rrf_chunks[hyde_chunk.id]
                total_score = hyde_score + existing_score
                rrf_chunks[hyde_chunk.id] = (existing_chunk, total_score)
            else:
                rrf_chunks[hyde_chunk.id] = (hyde_chunk, hyde_score)
        # 3. 取综合得分前10名
        merge_chunks = [
                           chunk for chunk, _ in sorted(rrf_chunks.values(), key=lambda item: item[1], reverse=True)
                       ][:10]
    except Exception as e:
        writer(QueryGraphStepInfo(name="RRF排序", status="failed", error=str(e)))
        return {"should_continue": False, "error": str(e)}

    logger.info(f"merge_chunks: {merge_chunks}")
    writer(QueryGraphStepInfo(name="RRF排序", status="success"))
    return {"rrf_chunks": merge_chunks}
