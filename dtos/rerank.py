from pydantic import BaseModel, Field
from typing import Literal


class ReRankChunk(BaseModel):
    title: str
    content: str
    url: str | None = None  # 仅网页搜索使用
    file_name: str | None = None
    file_url: str | None = None  # 仅向量搜索使用
    chunk_id: int | None = None
    source: Literal["web", "embedding"]


class ReRankedChunk(ReRankChunk):
    score: float = Field(..., description="重排序模型计算得分")