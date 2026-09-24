import json
from langgraph.runtime import Runtime
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo, QueryWebSearchChunk
from conf import app_config
from core.log import logger


async def web_search(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(QueryGraphStepInfo(name="联网搜索", status="running"))

    try:
        query = state.rewritten_query

        client = MultiServerMCPClient(
            {
                # 结构：{服务器名: {连接配置}}
                "web_search": {
                    "transport": "http",
                    "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
                    "headers": {"Authorization": f"Bearer {app_config.bailian.api_key}"},
                },
            }
        )
        tools = await client.get_tools()
        # print(tools)
        web_chunks: list[QueryWebSearchChunk] = []
        for tool in tools:
            if tool.name == "bailian_web_search":
                search_results = await tool.ainvoke({"query": query})
                search_result = search_results[0]
                pages = json.loads(search_result['text']).get("pages")
                for page in pages:
                    title = page.get("title")
                    content = page.get("snippet")
                    url = page.get("url")
                    web_chunks.append(
                        QueryWebSearchChunk(
                            title=title,
                            content=content,
                            url=url
                        )
                    )
        logger.info(f"web chunks: {web_chunks}")

        writer(QueryGraphStepInfo(name="web_search", status="success"))
        return {"web_chunks": web_chunks}
    except Exception as e:
        writer(QueryGraphStepInfo(name="web_search", status="failed", error=str(e)))
        return {"should_continue": False}
