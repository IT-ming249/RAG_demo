from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from apis.ingest_router import router as ingest_router
from apis.query_router import router as query_router
from clients.postgre import postgre_client
from repositories.milvus_repository import MilvusEntityRepository, MilvusChunkRepository
from clients.milvus import milvus_client


@asynccontextmanager
async def lifespan(_: FastAPI):
    await postgre_client.init()
    chunk_repo = MilvusChunkRepository(milvus_client.client)
    entity_repo = MilvusEntityRepository(milvus_client.client)
    await chunk_repo.ensure_collection()
    await entity_repo.ensure_collection()
    yield  # yield之前是项目启动前要的行动，在这之后是项目停止前的行动
    await postgre_client.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(ingest_router, prefix="/api/ingest", tags=["文档提取"])
app.include_router(query_router, prefix="/api/query", tags=["搜索"])

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, loop="asyncio", port=15236)
