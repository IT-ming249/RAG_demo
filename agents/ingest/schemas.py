# 存放信息提取相关的graph state
from pydantic import BaseModel, field_validator, ConfigDict
from typing import Literal
from pathlib import Path

from repositories.milvus_repository import MilvusEntityRepository, MilvusChunkRepository


class IngestMarkdownChunk(BaseModel):
    file_name: str
    title: str
    content: str
    header_chunk_index: int
    # 本块实际包含的实体名（多个用逗号拼接），由实体识别节点回填
    entity_name: str = ""


class IngestGraphState(BaseModel):
    file_path: Path
    markdown_dir: Path
    markdown_file: Path | None = None
    markdown_content: str | None = None
    markdown_chunks: list[IngestMarkdownChunk] | None = None
    entity_name: str | None = None

    should_continue: bool = True
    error: str | None = None


# Milvus对象上下文
class IngestGraphContext(BaseModel):
    milvus_entity_repository: MilvusEntityRepository
    milvus_chunk_repository: MilvusChunkRepository

    # 模型在实例化时，允许从对象属性读取数据，而不只是从字典读取↓
    model_config = ConfigDict(arbitrary_types_allowed=True)


# graph中间状态记录, 用于调试
class IngestGraphStepInfo(BaseModel):
    name: str
    status: Literal['running', 'success', 'failed']
    error: str | None = None
