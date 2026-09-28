from pydantic import BaseModel, Field
from typing import Literal


class ReRankChunk(BaseModel):
    title: str
    content: str
    url: str | None = None
    file_name: str | None = None
    chunk_id: int | None = None
    source: Literal["web", "embedding"]


class ReRankedChunk(ReRankChunk):
    score: float = Field(..., description="重排序模型计算得分")