import dashscope
from conf import app_config
from typing import Literal
from asgiref.sync import sync_to_async
from core.log import logger


async def generate_texts_embeddings(
        texts: list[str],
        output_type: Literal['dense', 'sparse', 'dense&sparse'] = "dense&sparse"
) -> list[dict[str, list[float] | dict[int, float]]] | None:
    try:
        response = await sync_to_async(dashscope.TextEmbedding.call)(
            model="qwen3.7-text-embedding",
            input=texts,
            api_key=app_config.bailian.api_key,
            output_type=output_type
        )
        embeddings = response['output']['embeddings']
        result = []
        for embedding in embeddings:
            dense_embeddings = embedding.get('embedding')
            raw_sparse_embeddings = embedding.get('sparse_embedding')
            # 先初始化，避免服务端未返回 sparse_embedding 时抛 UnboundLocalError
            sparse_embeddings: dict[int, float] = {}
            if raw_sparse_embeddings:
                sparse_embeddings = {e['index']: e['value'] for e in raw_sparse_embeddings}

            if output_type == "dense":
                result.append({"dense": dense_embeddings})
            elif output_type == "sparse":
                result.append({"sparse": sparse_embeddings})
            else:
                if dense_embeddings is None or not sparse_embeddings:
                    raise RuntimeError(
                        f"embedding 返回不完整: dense={'有' if dense_embeddings else '无'}, "
                        f"sparse={len(sparse_embeddings)} 项"
                    )
                result.append({"dense": dense_embeddings, "sparse": sparse_embeddings})
        return result
    except Exception as e:
        logger.error(str(e))
        return None
