from langgraph.runtime import Runtime

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from integrations.embedding import generate_texts_embeddings
from core.log import logger


async def embedding_search(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(QueryGraphStepInfo(name="搜索向量数据库", status="running"))

    contex = runtime.context

    query = state.rewritten_query
    entities = state.entities

    # 1. 生成查询问题向量
    assert query is not None
    query_embeddings = await generate_texts_embeddings([query])
    assert query_embeddings is not None
    dense_vector = query_embeddings[0].get("dense")
    sparse_vector = query_embeddings[0].get("sparse")

    # 2. 向量搜索问题相关的chunks
    milvus_chunk_repository = contex.milvus_chunk_repository
    chunks = await milvus_chunk_repository.search_chunks(
        [dense_vector],
        [sparse_vector],
        [entity.entity_name for entity in entities],
    )
    # logger.info(chunks)

    writer(QueryGraphStepInfo(name="搜索向量数据库", status="success"))
    return {"embedding_chunks": chunks}

