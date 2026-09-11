import re
import base64
import aiofiles
from pathlib import Path
from langgraph.runtime import Runtime
from langchain_core.prompts import PromptTemplate
from langchain.messages import HumanMessage

from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from agents.prompts import load_prompt
from clients.minio_client import minio_client
from agents.llm import vlm_client



async def image_to_base64(image_path: Path) -> str:
    async with aiofiles.open(image_path, mode="rb") as f:
        image_data = await f.read()
    b64str = base64.b64encode(image_data).decode("utf-8")
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(image_path.suffix.lower(), "application/octet-stream")
    return f"data:{mime_type};base64,{b64str}"


async def md_img_handle(state: IngestGraphState, runtime: Runtime[IngestGraphStepInfo]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="md文件图片处理", status="running"))

    md_file = state.markdown_file
    if not md_file:
        writer(IngestGraphStepInfo(name="md文件图片处理", status="running", error="md文件不存在"))
        return {"should_continue": False, "error": "md文件不存在"}

    try:
        async with aiofiles.open(state.markdown_file, "r", encoding='utf-8') as f:
            md_content = await f.read()

        image_dir = state.markdown_dir / "images"
        # md文件里面没图片的就可以不用处理了，直接返回
        if not image_dir.exists():
            return

        # 加载提示词模板
        raw_prompt = await load_prompt("summarize_image")
        image_handle_prompt_template = PromptTemplate(
            template=raw_prompt,
            input_variables=["pre_content", "next_content"]
        )

        for image_path in image_dir.glob("*"):
            if not image_path.is_file():
                continue

            # 1.将图片上传到minio返回图片url, 并将minerU解析出来的图片占位换成url
            image_url = await minio_client.upload_file(image_path, image_path.name)
            # 通过正则表达式替换 图片占位形式![](images/6d79be31310056eee41671f35f756ecc883e3528ee4e6a26189a8065a643d8ac.jpg)
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_path.name) + r".*?\)")

            # 2.将图片与上下文内容发给视觉模型，生成图片描述
            content_length = 100
            for match in pattern.finditer(md_content):
                start = match.start()
                end = match.end()
                pre_content = md_content[max(0, start - content_length):start]
                next_content = md_content[end:min(len(md_content), end + content_length)]
                image_base64 = await image_to_base64(image_path)
                # 视觉模型只能识别base64或公网url形式的图片
                image_handle_prompt = image_handle_prompt_template.format(
                    pre_content=pre_content, next_content=next_content)
                # 传递给视觉模型的标准message写法↓
                messages = [
                    HumanMessage(content=[
                        {"type": "text", "text": image_handle_prompt},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]),
                ]
                response = await vlm_client.ainvoke(messages)
                summary = response.content.strip().replace("\n", "")
                md_content = md_content = pattern.sub(f"![{summary}]({image_url})", md_content, count=1)

                # 3. 将处理后的markdown内容写回文件
                # 这一步仅测试用
                async with aiofiles.open(state.markdown_dir / "new_md.md", 'w', encoding='utf-8') as f:
                    await f.write(md_content)

    except Exception as e:
        writer(IngestGraphStepInfo(name="md文件图片处理", status="failed", error=str(e)))
        return {"should_continue": False, "error": str(e)}
    writer(IngestGraphStepInfo(name="md文件图片处理", status="success"))
    return {"markdown_content": md_content}
