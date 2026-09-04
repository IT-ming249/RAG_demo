# 存放信息提取相关的graph state
from pydantic import BaseModel, field_validator, ConfigDict
from typing import Literal
from pathlib import Path

class IngestGraphState(BaseModel):
    file_path: str
    markdown_dir: Path

    should_continue: bool = True
    error: str | None = None


# Minio, Miluvs对象上下文
class IngestGraphContext(BaseModel):
    pass

# graph中间状态记录, 用于调试
class IngestGraphStepInfo(BaseModel):
    name: str
    status: Literal['running', 'success', 'failed']
    error: str | None = None