from pydantic import BaseModel


class MilvusInsertEntity(BaseModel):
    file_name: str
    entity_name: str
    dense_vector: list[float] | None = None
    # 稀疏向量结构：{索引， 值}
    sparse_vector: dict[int, float] | None = None
