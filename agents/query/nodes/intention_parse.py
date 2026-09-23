from langgraph.runtime import Runtime
from langchain_core.messages import BaseMessage

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from agents.ainvoke_llm import ainvoke_llm_json
from core.log import logger


async def intention_parse(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(QueryGraphStepInfo(name="意图识别", status="running"))

    try:
        # 历史消息，不包含最近的一条
        history_messages: list[BaseMessage] = state.messages[0:-1]
        result = await ainvoke_llm_json(
            "intention_parse",
            {
                "messages": [{"role": message.type, "content": message.content}
                             for message in history_messages],
                "query": state.query,
            }
        )
    except Exception as e:
        writer(QueryGraphStepInfo(name="意图识别", status="failed", error=str(e)))
        return {"should_continue": False}
    logger.info(f"识别结果：{result["should_continue"]}")
    writer(QueryGraphStepInfo(name="意图识别", status="success"))
    return {"should_continue": result["should_continue"]}
