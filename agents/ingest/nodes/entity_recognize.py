from langgraph.runtime import Runtime

from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from agents.ainovke_llm import ainvoke_llm_str
from integrations.embedding import generate_texts_embeddings
from repositories.milvus_repository import MilvusEntityRepository
from dots.miluvs import MilvusInsertEntity
from core.log import logger


# 实体输出中的分隔符（全角竖线，与 recognize_entity.md 提示词一致）
ENTITY_TYPE_SEP = "｜"


def _extract_entity_names(entity: str) -> list[str]:
    """
    从实体行中提取用于文本匹配的名字。

    支持两种格式：
      - 有别名: "陆地卫星=Landsat｜PROGRAM" -> ["陆地卫星", "Landsat"]
      - 无别名: "NASA｜ORG"                -> ["NASA"]
    """
    # 去掉类型后缀
    body = entity.split(ENTITY_TYPE_SEP, 1)[0].strip()
    if not body:
        return []
    # 别名与规范名都参与匹配
    return [n.strip() for n in body.split("=") if n.strip()]


async def entity_recognize(state: IngestGraphState, runtime: Runtime[IngestGraphStepInfo]):
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

        # 5. 回填每个chunk实际包含的实体名（多个用逗号拼接）
        for chunk in state.markdown_chunks:
            chunk_text = f"{chunk.title}\n{chunk.content}"
            matched = []
            for entity in entities:
                # 从 "别名=规范名｜类型" 或 "规范名｜类型" 中提取用于匹配的名字
                names = _extract_entity_names(entity)
                if any(name and name in chunk_text for name in names):
                    matched.append(entity)
            # 去重后保持原顺序拼接
            chunk.entity_name = ",".join(dict.fromkeys(matched))
    except Exception as e:
        writer(IngestGraphStepInfo(name="实体识别", status="failed", error=str(e)))
        return {"should_continue": False}
    logger.info(f"entity_recognize: 共识别 {len(entities)} 个实体: {entities}")

    writer(IngestGraphStepInfo(name="实体识别", status="success"))
    return {"entity_name": entity_names, "markdown_chunks": state.markdown_chunks}



