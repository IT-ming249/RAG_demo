from langgraph.runtime import Runtime

from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from agents.ainvoke_llm import ainvoke_llm_str
from integrations.embedding import generate_texts_embeddings
from repositories.milvus_repository import MilvusEntityRepository
from dtos.milvus import MilvusInsertEntity
from core.log import logger


async def entity_recognize(state: IngestGraphState, runtime: Runtime[IngestGraphContext]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="实体识别", status="running"))

    try:
        context = runtime.context

        file_name = state.markdown_file.name

        # 1. 从markdown_chunks取出前几块内容（避免输入过长）做实体识别
        top_chunks = state.markdown_chunks[:5]
        content = "\n\n".join([f"{chunk.title}\n{chunk.content}" for chunk in top_chunks])

        # 2. 用大模型识别实体
        entity_names = await ainvoke_llm_str("recognize_entity", {'file_name': file_name, 'context': content})
        # 按行拆实体
        entities = [ln.strip() for ln in entity_names.splitlines() if ln.strip()]

        # 3. 生成实体向量（批量：一次请求为所有实体生成向量，返回顺序与 entities 一致）
        embedding_results = await generate_texts_embeddings(entities)
        assert embedding_results is not None

        # 4. 把实体以及向量逐个添加到milvus中
        milvus_entity_repo: MilvusEntityRepository = context.milvus_entity_repository
        for entity, embedding in zip(entities, embedding_results):
            await milvus_entity_repo.add_entity(
                MilvusInsertEntity(
                    file_name=file_name,
                    entity_name=entity,
                    dense_vector=embedding.get('dense'),
                    sparse_vector=embedding.get('sparse')
                )
            )

        # 5. 回填每个chunk所属的产品名
        #    提示词要求只提取整份文档最主要的那一个产品，因此文档内所有chunk都归属于它，
        #    无需逐块做字符串匹配（块内可能用"该设备"等指代，匹配会大量落空）。
        primary_product = entities[0] if entities else ""
        for chunk in state.markdown_chunks:
            chunk.entity_name = primary_product
    except Exception as e:
        writer(IngestGraphStepInfo(name="实体识别", status="failed", error=str(e)))
        return {"should_continue": False}
    logger.info(f"entity_recognize: 识别出产品 {len(entities)} 个: {entities}")

    writer(IngestGraphStepInfo(name="实体识别", status="success"))
    return {"entity_name": entity_names, "markdown_chunks": state.markdown_chunks}



