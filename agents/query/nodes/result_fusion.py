from langgraph.runtime import Runtime

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from agents.ainvoke_llm import astream_llm_str
from agents.query.stream_events import step_event


async def result_fusion(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="结果融合", status="running")))
