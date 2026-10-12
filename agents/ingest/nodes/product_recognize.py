from langgraph.runtime import Runtime

from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from agents.ainvoke_llm import ainvoke_llm_str
from integrations.embedding import generate_texts_embeddings
from repositories.milvus_repository import MilvusProductRepository
from dtos.milvus import MilvusInsertProduct
from core.log import logger


async def product_recognize(state: IngestGraphState, runtime: Runtime[IngestGraphContext]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="产品识别", status="running"))

    try:
        context = runtime.context

        file_name = state.markdown_file.name

        # 1. 从markdown_chunks取出前几块内容（避免输入过长）做产品识别
        top_chunks = state.markdown_chunks[:5]
        content = "\n\n".join([f"{chunk.title}\n{chunk.content}" for chunk in top_chunks])

        # 2. 用大模型识别产品
        product_names = await ainvoke_llm_str("recognize_product", {'file_name': file_name, 'context': content})
        # 按行拆产品
        products = [ln.strip() for ln in product_names.splitlines() if ln.strip()]

        # 3. 生成产品向量（批量：一次请求为所有产品生成向量，返回顺序与 products 一致）
        embedding_results = await generate_texts_embeddings(products)
        assert embedding_results is not None

        # 4. 把产品以及向量逐个添加到milvus中
        milvus_product_repo: MilvusProductRepository = context.milvus_product_repository
        for product, embedding in zip(products, embedding_results):
            await milvus_product_repo.add_product(
                MilvusInsertProduct(
                    file_name=file_name,
                    product_name=product,
                    dense_vector=embedding.get('dense'),
                    sparse_vector=embedding.get('sparse')
                )
            )

        # 5. 回填每个chunk所属的产品名
        #    提示词要求只提取整份文档最主要的那一个产品，因此文档内所有chunk都归属于它，
        #    无需逐块做字符串匹配（块内可能用"该设备"等指代，匹配会大量落空）。
        primary_product = products[0] if products else ""
        for chunk in state.markdown_chunks:
            chunk.product_name = primary_product
    except Exception as e:
        writer(IngestGraphStepInfo(name="产品识别", status="failed", error=str(e)))
        return {"should_continue": False}
    logger.info(f"product_recognize: 识别出产品 {len(products)} 个: {products}")

    writer(IngestGraphStepInfo(name="产品识别", status="success"))
    return {"product_name": product_names, "markdown_chunks": state.markdown_chunks}



