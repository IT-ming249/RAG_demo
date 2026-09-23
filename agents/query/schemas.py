from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Literal

from repositories.milvus_repository import MilvusChunkRepository, MilvusEntityRepository
from dtos.milvus import MilvusSearchEntity, MilvusSearchChunk


class QueryGraphState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    rewritten_query: str | None = None
    entities: list[MilvusSearchEntity] | None = None
    embedding_chunks: list[MilvusSearchChunk] | None = None
    hyde_chunks: list[MilvusSearchChunk] | None = None
    should_continue: bool = True
    error: str | None = None


class QueryGraphContext(BaseModel):
    milvus_entity_repository: MilvusEntityRepository
    milvus_chunk_repository: MilvusChunkRepository

    model_config = ConfigDict(arbitrary_types_allowed=True)


class QueryGraphStepInfo(BaseModel):
    name: str
    status: Literal['running', 'success', 'failed']
    error: str | None = None
