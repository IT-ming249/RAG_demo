from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from langgraph.runtime import Runtime



async def md_split(state: IngestGraphState, runtime: Runtime[IngestGraphStepInfo]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="md文档分割", status="running"))