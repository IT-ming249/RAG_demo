from langgraph.runtime import Runtime

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from agents.ainvoke_llm import ainvoke_llm_str
from integrations.embedding import generate_texts_embeddings
from core.log import logger


async def hyde_search(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(QueryGraphStepInfo(name="假性文档检索", status="running"))

    contex = runtime.context

    query = state.rewritten_query
    entities = state.entities

    # 1. 大模型生成hyde
    hyde_result = await ainvoke_llm_str("hyde_generate", {"query": query})

    # 2. hyde向量化
    hyde_embedding = await generate_texts_embeddings([hyde_result])
    assert hyde_embedding is not None
    hyde_dense_vector = hyde_embedding[0].get("dense")
    hyde_sparse_vector = hyde_embedding[0].get("sparse")

    # 3. hyde向量搜索
    milvus_chunk_repository = contex.milvus_chunk_repository
    chunks = await milvus_chunk_repository.search_chunks(
        [hyde_dense_vector],
        [hyde_sparse_vector],
        entity_names=[entity.entity_name for entity in entities]
    )

    logger.info(f"Hyde:{hyde_result}")
    logger.info(f"Hyde search results: {chunks}")

    writer(QueryGraphStepInfo(name="假设性文档检索", status="success"))
    return {"hyde_chunks": chunks}


