from langgraph.runtime import Runtime
from langchain_core.messages import BaseMessage
from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo


async def entity_confirm(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(QueryGraphStepInfo(name="实体识别", status="running"))

    context = runtime.context
    messages = state.messages

    history_messages = list(BaseMessage)
