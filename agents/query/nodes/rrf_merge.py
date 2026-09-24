from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from langgraph.runtime import Runtime


async def rrf_merge(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(QueryGraphStepInfo(name="RRF排序", status="running"))

    embedding_chunks = state.embedding_chunks
    hyde_chunks = state.hyde_chunks

    assert embedding_chunks is not None
    assert  hyde_chunks is not None
