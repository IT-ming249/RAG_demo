from fastapi import FastAPI
from apis.ingest_router import router as ingest_router
import uvicorn

app = FastAPI()
app.include_router(ingest_router, prefix="/api/ingest", tags=["文档提取"])


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, loop="asyncio", port=15236)
