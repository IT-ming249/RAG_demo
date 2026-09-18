import json
from pymilvus import AsyncMilvusClient, DataType, AnnSearchRequest, WeightedRanker

from dots.miluvs import MilvusInsertEntity, MilvusSearchEntity, MilvusInsertChunk, MilvusSearchChunk
from utils.l2_normalize import l2_normalize


class MilvusEntityRepository:
    collection_name = "entity_collection"

    def __init__(self, client: AsyncMilvusClient):
        self.client = client

    async def clear_collection(self):
        has_collection = await self.client.has_collection(self.collection_name)
        if has_collection:
            await self.client.drop_collection(self.collection_name)

    async def ensure_collection(self):
        has_collection = await self.client.has_collection(self.collection_name)
        if has_collection:
            return
        # 如果没用就创建
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 主键
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 文件名
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=255)
        # 实体名
        schema.add_field(field_name="entity_name", datatype=DataType.VARCHAR, max_length=65535)
        # 稠密向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 稀疏向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 构建索引
        index_params = self.client.prepare_index_params()
        # 为稠密向量构建索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",  # 索引算法
            metric_type="COSINE",  # 度量算法
            params={"M": 16, "efConstruction": 200}
        )
        """
        ● M：在构建索引时，每个节点（向量）在图中最多拥有的双向链接数量。M越大，图的结构越密集，召回率（Recall）越高，但索引体积会显著增大（内存占用增加），且构建时间变长。通常取值范围在 4 ~ 64。16 是官方推荐的默认值，适合大多数场景。如果内存紧张，可以调低至 8；如果追求极致精度且内存充裕，可以设为 32。
        ● efConstruction：在构建图的过程中，用于搜索最近邻的动态候选列表大小。这个值越大，构建时搜索的邻居范围越广，索引质量越高（召回率越高），但构建时间会显著变长。通常取值范围在 100 ~ 500。200 是均衡值。注意：efConstruction 不影响 索引大小，只影响构建速度和最终精度。如果数据量巨大且对查询精度要求极高，可以设为 400，但建索引时间会翻倍。
        """
        # 为稀疏向量构建索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            # IP：内积
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )
        """
        ● inverted_index_algo：决定在查询时如何遍历倒排列表（Term 对应的文档 ID 列表）。DAAT_MAXSCORE是 Milvus 最推荐的算法。它代表“文档有序遍历 + MaxScore 剪枝”。它利用稀疏向量的特性（大量 Term 权重低），在遍历时跳过低分文档，能在保证高召回率的同时大幅提升查询速度。另一个选项是 DAAT_TAAT（Term 有序遍历），速度较慢，一般不建议使用。
        ● normalize：是否在构建索引时对向量进行 L2 归一化（让向量模长为 1）。在归一化后，两个向量的内积等于它们的余弦相似度。对于文本稀疏向量，词频差异很大，归一化可以消除文档长度对相似度计算的影响（即只关注方向，不关注长度），这符合大部分文本检索的需求。注意，这里只是在建立索引的时候进行归一化，在查询前需要手动将查询向量归一化后再进行查询。
        ● quantization：量化策略。对浮点数（权重）进行压缩，减少内存占用。设为 "none"表示不进行量化，保持原始 float32 精度，这会消耗更多内存，但能保证最高的召回率。如果你内存紧张，可以改为 "int8" 或 "fp16" 来压缩，但会牺牲微小的精度。
        """

        await self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                            index_params=index_params)

    async def add_entity(self, entity: MilvusInsertEntity):
        # 先删除同名实体
        await self.client.delete(
            self.collection_name,
            filter=f'file_name == "{entity.file_name}" && entity_name == "{entity.entity_name}"'
        )
        # 对稀疏向量做归一化处理
        entity.sparse_vector = {
            k: v
            for k, v in zip(entity.sparse_vector.keys(), l2_normalize(list(entity.sparse_vector.values())))
        }
        # 将entity添加到Milvus中
        await self.client.insert(self.collection_name, data=[entity.model_dump()])
        # 强制加载
        await self.client.load_collection(self.collection_name)

    async def search_entity(self,
                            dense_vectors: list[list[float]],
                            sparse_vectors: list[dict[int, float]],
                            limit=5) -> list[MilvusSearchEntity]:
        # 构建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=dense_vectors,
            anns_field="dense_vector",  # 向量索引构建时的field_name
            param={"metric_type": "COSINE"},  # 度量算法
            limit=limit
        )
        sparse_request = AnnSearchRequest(
            data=sparse_vectors,
            anns_field="sparse_vector",
            param={"metric_type": "IP"}, limit=limit
        )

        # 0.8 0.2 分别对应分配给第一条/第二条request的权重，在这里对应稠密向量0.8，稀疏向量0.2
        ranker = WeightedRanker(0.8, 0.2, norm_score=True)
        result = await self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=limit,
            output_fields=['entity_name']
        )

        items: list[MilvusSearchEntity] = []
        for entities in result:
            for entity in entities:
                items.append(MilvusSearchEntity(
                    id=entity['id'],
                    distance=entity['distance'],
                    entity_name=entity['entity']['entity_name'],
                ))
        return items


class MilvusChunkRepository:
    collection_name = "chunks_collection"

    def __init__(self, client: AsyncMilvusClient):
        self.client = client

    async def ensure_collection(self):
        has_collection = await self.client.has_collection(self.collection_name)
        if has_collection:
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 主键
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 文件名
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=255)
        # 标题
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        # 内容
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        # chunk索引号
        schema.add_field(field_name="header_chunk_index", datatype=DataType.INT8)
        # entity_name
        schema.add_field(field_name="entity_name", datatype=DataType.VARCHAR, max_length=65535)
        # 稠密向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 稀疏向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 创建索引
        index_params = self.client.prepare_index_params()
        # 为稠密向量构建索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        # 为稀疏向量构建索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            # IP：内积
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        await self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                            index_params=index_params)

    async def clear_collection(self):
        has_collection = await self.client.has_collection(self.collection_name)
        if has_collection:
            await self.client.drop_collection(self.collection_name)

    async def add_chunks(self, chunks: list[MilvusInsertChunk]):
        # 批量插入chunks
        chunk_size: int = 10
        for index in range(0, len(chunks), chunk_size):
            batch_chunks = chunks[index:index + chunk_size]
            chunk_dicts = [chunk.model_dump() for chunk in batch_chunks]
            await self.client.insert(collection_name=self.collection_name, data=chunk_dicts)
        await self.client.load_collection(self.collection_name)

    async def search_chunks(self,
                            dense_vectors: list[list[float]],
                            sparse_vectors: list[dict[int, float]],
                            entity_names: list[str],
                            limit: int = 10
                            ):
        expr = f'entity_name in {json.dumps(entity_names, ensure_ascii=False)}'
        # 构建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=dense_vectors,
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            expr=expr,
            limit=limit
        )
        # 构建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=sparse_vectors,
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            expr=expr,
            limit=limit
        )

        ranker = WeightedRanker(0.5, 0.5, norm_score=True)
        result = await self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=limit,
            output_fields=['id', "title", "content", "entity_name", "file_name"]
        )
        items: list[MilvusSearchChunk] = []
        for chunks in result:
            for chunk in chunks:
                items.append(
                    MilvusSearchChunk(
                        id=chunk['id'],
                        distance=chunk['distance'],
                        title=chunk['entity']['title'],
                        content=chunk['entity']['content'],
                        entity_name=chunk['entity']['entity_name'],
                        file_name=chunk['entity']['file_name']
                    )
                )
        return items
