from langgraph.runtime import Runtime
from langchain_core.messages import BaseMessage, AIMessage

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from agents.ainvoke_llm import ainvoke_llm_json
from agents.query.stream_events import step_event
from integrations.embedding import generate_texts_embeddings
from core.log import logger


async def product_confirm(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="产品识别", status="running")))

    context = runtime.context
    messages = state.messages

    try:
        # 1.调用大模型识别用户对话中提到的产品，并重新组织查询语句
        history_messages: list[BaseMessage] = messages[0:-1]
        result = await ainvoke_llm_json(
            "product_parse",
            {
                "history_text": [{"role": message.type, "content": message.content}
                                 for message in history_messages],
                "query": state.query,
            }
        )

        product_names: list[str] = result["product_names"]
        rewritten_query: str = result["rewritten_query"]

        if len(product_names) == 0:
            error = "无法从用户描述中识别到合适的产品名称"
            writer(step_event(QueryGraphStepInfo(name="产品识别", status="failed", error=error)))
            return {"should_continue": False, "error": error}
        logger.info(f"提取的产品名称: {product_names}， 重写查询: {rewritten_query}")

        # 2. 将用户对话中识别的产品转换为向量
        product_embeddings = await generate_texts_embeddings(product_names)
        assert product_embeddings is not None

        dense_vectors: list[list[float]] = []
        sparse_vectors: list[dict[int, float]] = []
        for product_embedding in product_embeddings:
            dense_vectors.append(product_embedding.get("dense"))
            sparse_vectors.append(product_embedding.get("sparse"))

        # 3. 向量数据库中搜索相关产品, 并去重
        milvus_product_repository = context.milvus_product_repository
        products = await milvus_product_repository.search_product(dense_vectors, sparse_vectors)
        # 去重
        seen = set()
        for item in products[:]:
            if item.product_name in seen:
                # 循环中直接删掉迭代元素会影响循环，所以↑遍历的对象是个完整切片
                products.remove(item)
            else:
                seen.add(item.product_name)

        # 按照distance倒序排序， 当前搜索采用的是cosine与IP算法，所以distance越大相关性越大
        products.sort(key=lambda product: product.distance, reverse=True)
        # 筛选
        products = list(filter(lambda product: product.distance > 0.65, products))

        # logger.info(f"查找到的产品有：{products}")

        if len(products) == 0:
            writer(step_event(QueryGraphStepInfo(name="产品识别", status="failed", error="未找到合适产品")))
            return {"messages": [AIMessage(content="未找到合适产品")], "should_continue": False}
        else:
            content = f"查找到的产品名称为{[item.model_dump() for item in products]}，改写后的问题为：{rewritten_query}"
            writer(step_event(QueryGraphStepInfo(name="产品识别", status="success")))
            return {"messages": [AIMessage(content=content)], "products": products, "rewritten_query": rewritten_query}
    except Exception as e:
        writer(step_event(QueryGraphStepInfo(name="产品识别", status="failed", error=str(e))))
        return {"should_continue": False}





