import tempfile
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
import json

from core.log import logger
from clients.minio_client import minio_client
from agents.ingest.graph import graph
from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from repositories.milvus_repository import MilvusEntityRepository, MilvusChunkRepository
from clients.milvus import milvus_client


router = APIRouter()


async def _event_stream(
    original_name: str,
    local_path: Path,
    file_url: str,
    object_name: str,
    markdown_dir: Path
):
    """
    SSE事件生成器
    """
    try:
        state = IngestGraphState(
            file_url=file_url,
            file_path=local_path,
            markdown_dir=markdown_dir
        )
        context = IngestGraphContext(
            milvus_entity_repository=MilvusEntityRepository(milvus_client.client),
            milvus_chunk_repository=MilvusChunkRepository(milvus_client.client)
        )
        async for chunk in graph.astream(state, context=context, stream_mode="custom"):
            if isinstance(chunk, IngestGraphStepInfo):
                yield f"event: step\ndata: {chunk.model_dump_json(ensure_ascii=False)}\n\n"
                # 发送完成事件
        yield f"event: done\ndata: {json.dumps({'file_name': original_name, 'object_name': object_name}, 
                                               ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.error(f"执行文档提取agent失败： {str(e)}")
        yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传PDF文件, 并向量化存储
    """
    # 1. 上传文件的object_name
    original_name = Path(file.filename or "unknown.pdf").name
    file_id = str(uuid.uuid4().hex)
    object_name = f"pdfs/{file_id}/{original_name}"

    # 2. 将pdf存储到服务器临时文件目录 + 上传到MinIO
    temp_dir = Path(tempfile.mkdtemp())
    local_path = temp_dir / "RAG_TEMP" / original_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_dir = temp_dir / "RAG_MARKDOWN_TEMP"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with aiofiles.open(local_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        # 上传到MinIo
        file_url = await minio_client.upload_file(str(local_path.resolve()), object_name)
        logger.info(f"上传文件url: {file_url}")
    except Exception as e:
        logger.error(f"文件上传失败：{e}")
        raise

    # 流式返回 ingest agent 执行情况
    return StreamingResponse(
            _event_stream(original_name, local_path, file_url, object_name, markdown_dir),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )


