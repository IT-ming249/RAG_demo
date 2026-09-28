from langgraph.runtime import Runtime

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from agents.query.stream_events import step_event
from dtos.rerank import ReRankChunk
from integrations.rerank import rerank_documents
from core.log import logger


async def chunks_rerank(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="重排序", status="running")))

    rrf_chunks = state.rrf_chunks
    web_chunks = state.web_chunks
    assert rrf_chunks is not None
    assert web_chunks is not None

    # 1. 合并所有的来源的chunks, 为重排序做数据准备
    merged_chunks: list[ReRankChunk] = []
    for rrf_chunk in rrf_chunks:
        merged_chunks.append(ReRankChunk(
            title=rrf_chunk.title,
            content=rrf_chunk.content,
            file_name=rrf_chunk.file_name,
            chunk_id=rrf_chunk.id,
            source="embedding"
        ))

    for web_chunk in web_chunks:
        merged_chunks.append(ReRankChunk(
            title=web_chunk.title,
            content=web_chunk.content,
            url=web_chunk.url,
            source="web"
        ))

    # 2. 对合并后的chunk进行重排序
    try:
        reranked_chunks = await rerank_documents(state.rewritten_query, merged_chunks)
        # logger.info(f"reranked chunks: {reranked_chunks}")
        assert reranked_chunks is not None
    except Exception as e:
        writer(step_event(QueryGraphStepInfo(name="重排序", status="failed", error=str(e))))
        return {"should_continue": False, "error": str(e)}

    # 3. 取TopK：结合固定上下限+断崖阈值判断，避免机械取前N条，保留语义相关的连续文档集合
    max_topk = 10
    min_topk = 1
    max_abs_gap = 0.5  # 最大分差绝对值
    max_ratio_gap = 0.25  # 最大分差比例

    k = 0
    for index in range(0, len(reranked_chunks) - 1):
        current_chunk = reranked_chunks[index]
        next_chunk = reranked_chunks[index + 1]
        # 绝对分差
        abs_gap = current_chunk.score - next_chunk.score
        # 相对分差
        ratio_gap = abs_gap / (current_chunk.score + 1e-6)
        if abs_gap > max_abs_gap:
            break
        elif ratio_gap > max_ratio_gap:
            break
        else:
            k += 1

    final_chunks = reranked_chunks[0:min(max(k, min_topk), max_topk)]
    # logger.info(f"final chunk: {final_chunks}")

    writer(step_event(QueryGraphStepInfo(name="重排序", status="success")))
    return {"reranked_chunks": final_chunks}
