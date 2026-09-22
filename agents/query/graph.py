from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agents.query.schemas import QueryGraphState, QueryGraphContext
from agents.query.nodes.intention_parse import intention_parse
from agents.query.nodes.entity_confirm import entity_confirm
from agents.query.nodes.embedding_search import embedding_search
from agents.query.nodes.hyde_search import hyde_search
from agents.query.nodes.rrf_merge import rrf_merge
from agents.query.nodes.web_search import web_search
from agents.query.nodes.result_fusion import result_fusion
from agents.query.nodes.chunks_rerank import chunks_rerank
from clients.milvus import milvus_client
from clients.postgre import postgre_client
from repositories.milvus_repository import MilvusChunkRepository, MilvusEntityRepository
from core.log import logger


def build_graph_builder() -> StateGraph:
    graph_builder = StateGraph(
        state_schema=QueryGraphState,
        context_schema=QueryGraphContext
    )
    graph_builder.add_node(intention_parse)
    graph_builder.add_node(entity_confirm)
    graph_builder.add_node(embedding_search)
    graph_builder.add_node(hyde_search)
    graph_builder.add_node(rrf_merge)
    graph_builder.add_node(web_search)
    graph_builder.add_node(chunks_rerank)
    graph_builder.add_node(result_fusion)

    graph_builder.add_edge(START, "intention_parse")
    graph_builder.add_conditional_edges(
        "intention_parse",
        lambda state: "entity_confirm" if state.should_continue else END
    )
    graph_builder.add_conditional_edges(
        "entity_confirm",
        lambda state: "embedding_search" if state.should_continue else END
    )
    graph_builder.add_conditional_edges(
        "entity_confirm",
        lambda state: "hyde_search" if state.should_continue else END
    )
    graph_builder.add_conditional_edges(
        "entity_confirm",
        lambda state: "web_search" if state.should_continue else END
    )

    # Wait for both retrieval branches before merging their RRF scores.
    graph_builder.add_edge(["embedding_search", "hyde_search", "web_search"], "rrf_merge")
    graph_builder.add_edge("rrf_merge", "chunks_rerank")
    graph_builder.add_edge("chunks_rerank", "result_fusion")
    graph_builder.add_edge("result_fusion", END)

    return graph_builder


async def get_graph():
    graph_builder = build_graph_builder()
    checkpointer = AsyncPostgresSaver(conn=postgre_client.pool)
    await checkpointer.setup()
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph
