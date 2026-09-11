# 存放信息提取相关的graph state
from pydantic import BaseModel, field_validator, ConfigDict
from typing import Literal
from pathlib import Path

class IngestGraphState(BaseModel):
    file_path: Path
    markdown_dir: Path
    markdown_file: Path | None = None
    markdown_content: str | None = None

    should_continue: bool = True
    error: str | None = None


# Minio, Milvus对象上下文
class IngestGraphContext(BaseModel):
    pass

# graph中间状态记录, 用于调试
class IngestGraphStepInfo(BaseModel):
    name: str
    status: Literal['running', 'success', 'failed']
    error: str | None = None