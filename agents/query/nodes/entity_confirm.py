from langgraph.runtime import Runtime
from langchain_core.messages import BaseMessage, AIMessage

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from agents.ainvoke_llm import ainvoke_llm_json
from integrations.embedding import generate_texts_embeddings
from core.log import logger


async def entity_confirm(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(QueryGraphStepInfo(name="实体识别", status="running"))

    context = runtime.context
    messages = state.messages

    # 1.调用大模型识别用户对话中提到的实体，并重新组织查询语句
    history_messages: list[BaseMessage] = messages[0:-1]
    result = await ainvoke_llm_json(
        "entity_parse",
        {
            "history_text": [{"role": message.type, "content": message.content}
                             for message in history_messages],
            "query": state.query,
        }
    )

    entity_names: list[str] = result["entity_names"]
    rewritten_query: str = result["rewritten_query"]

    if len(entity_names) == 0:
        error = "无法从用户描述中识别到合适的产品名称"
        writer(QueryGraphStepInfo(name="实体识别", status="failed", error=error))
        return {"should_continue": False, "error": error}
    logger.info(f"提取的实体名称: {entity_names}， 重写查询: {rewritten_query}")

    # 2. 将用户对话中识别的实体转换为向量
    entity_embeddings = await generate_texts_embeddings(entity_names)
    assert entity_embeddings is not None

    dense_vectors: list[list[float]] = []
    sparse_vectors: list[dict[int, float]] = []
    for entity_embedding in entity_embeddings:
        dense_vectors.append(entity_embedding.get("dense"))
        sparse_vectors.append(entity_embedding.get("sparse"))

    # 3. 向量数据库中搜索相关实体, 并去重
    milvus_entity_repository = context.milvus_entity_repository
    entities = await milvus_entity_repository.search_entity(dense_vectors, sparse_vectors)
    # 去重
    seen = set()
    for item in entities[:]:
        if item.entity_name in seen:
            # 循环中直接删掉迭代元素会影响循环，所以↑遍历的对象是个完整切片
            entities.remove(item)
        else:
            seen.add(item.entity_name)

    # 按照distance倒序排序， distance越大相关性越大
    entities.sort(key=lambda entity: entity.distance, reverse=True)
    # 筛选
    entities = list(filter(lambda entity: entity.distance > 0.65, entities))

    # logger.info(f"查找到的实体有：{entities}")

    if len(entities) == 0:
        writer(QueryGraphStepInfo(name="实体识别", status="failed", error="未找到合适产品"))
        return {"messages": [AIMessage(content="未找到合适产品")], "should_continue": False}
    else:
        content = f"查找到的产品名称为{[item.model_dump() for item in entities]}，改写后的问题为：{rewritten_query}"
        writer(QueryGraphStepInfo(name="entity_confirm", status="success"))
        return {"messages": [AIMessage(content=content)], "entities": entities, "rewritten_query": rewritten_query}





