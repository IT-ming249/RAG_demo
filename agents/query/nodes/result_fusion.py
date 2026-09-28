from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from agents.ainvoke_llm import astream_llm_str
from agents.query.stream_events import step_event, answer_delta_event
from core.log import logger


async def result_fusion(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="结果融合", status="running")))

    reranked_chunks = state.reranked_chunks
    assert reranked_chunks is not None
    messages = state.messages

    final_answer = ""
    async for delta in astream_llm_str(
        "result_fusion",
            {
                "chunks": [chunk.model_dump() for chunk in reranked_chunks],
                "messages": [{"role": message.type, "content": message.content} for message in messages],
                "entity_names": [entity.entity_name for entity in state.entities],
                "query": state.rewritten_query,
            }
    ):
        final_answer += delta
        writer(answer_delta_event(delta=delta))

    writer(step_event(QueryGraphStepInfo(name="结果融合", status="success")))

    logger.info(f"最终回复: {final_answer}")
    return {"messages": [AIMessage(content=final_answer)]}
