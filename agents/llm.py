from langchain_openai import ChatOpenAI
from conf import app_config
from pydantic import SecretStr

# 语言模型
llm = ChatOpenAI(
    model=app_config.llm.model_name,
    api_key=app_config.llm.api_key,
    base_url=app_config.base_url
)

# 多模态模型
vlm = ChatOpenAI(
    model=app_config.vlm.model_name,
    api_key=app_config.vlm.api_key,
    base_url=app_config.vlm.base_url
)