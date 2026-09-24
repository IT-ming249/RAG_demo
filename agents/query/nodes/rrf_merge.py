from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from langgraph.runtime import Runtime


async def rrf_merge(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(QueryGraphStepInfo(name="RRF排序", status="running"))

    embedding_chunks = state.embedding_chunks
    hyde_chunks = state.hyde_chunks

    assert embedding_chunks is not None
    assert hyde_chunks is not None

    try:
        # 1. 两路向量搜索结果按照得分排序
        embedding_chunks.sort(key=lambda chunk: chunk.distance, reverse=True)
        hyde_chunks.sort(key=lambda chunk: chunk.distance, reverse=True)


    except Exception as e:
        writer(QueryGraphStepInfo(name="RRF排序", status="failed", error=str(e)))
        return {"should_continue": False}
