from pydantic import BaseModel


class MilvusInsertProduct(BaseModel):
    file_name: str
    product_name: str
    dense_vector: list[float] | None = None
    # 稀疏向量结构：{索引， 值}
    sparse_vector: dict[int, float] | None = None


class MilvusSearchProduct(BaseModel):
    id: int
    distance: float
    product_name: str


class MilvusInsertChunk(BaseModel):
    file_url: str
    file_name: str
    title: str
    content: str
    header_chunk_index: int
    product_name: str
    dense_vector: list[float] | None = None
    sparse_vector: dict[int, float] | None = None


class MilvusSearchChunk(BaseModel):
    id: int
    distance: float
    title: str
    content: str
    product_name: str
    file_name: str
    file_url: str
