from prompts import load_prompt
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from llm import llm_client


async def ainvoke_llm_json(prompt_name: str, kwargs: dict):
    prompt = await load_prompt(prompt_name)
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm_client | JsonOutputParser()
    result = await chain.ainvoke(kwargs)
    return result

async def ainvoke_llm_str(prompt_name: str, kwargs: dict):
    prompt = await load_prompt(prompt_name)
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm_client | StrOutputParser()
    result = await chain.ainvoke(kwargs)
    return result


async def astream_llm_str(prompt_name: str, kwargs: dict):
    prompt = await load_prompt(prompt_name)
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm_client
    async for chunk in chain.astream(kwargs):
        """
        chunk.content可能的结构：
        chunk.content = "xxx"
        chunk.content = [
            {"type": "text", "text": "你好"},
            {"type": "text", "text": "，很高兴见到你"}
            "yyy"
        ]
        """
        content = getattr(chunk, "content", None)
        if not content:
            continue
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    yield part
                elif isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        yield text