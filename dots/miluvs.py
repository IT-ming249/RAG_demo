from pydantic import BaseModel


class MilvusInsertEntity(BaseModel):
    file_name: str
    entity_name: str
    dense_vector: list[float] | None = None
    # 稀疏向量结构：{索引， 值}
    sparse_vector: dict[int, float] | None = None


class MilvusSearchEntity(BaseModel):
    id: int
    distance: float
    entity_name: str


class MilvusInsertChunk(BaseModel):
    file_name: str
    title: str
    content: str
    header_chunk_index: int
    entity_name: str
    dense_vector: list[float] | None = None
    sparse_vector: dict[int, float] | None = None


class MilvusSearchChunk(BaseModel):
    id: int
    distance: float
    title: str
    content: str
    entity_name: str
    file_name: str
