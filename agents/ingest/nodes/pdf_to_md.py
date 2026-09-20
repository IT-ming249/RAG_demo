from agents.ingest.schemas import IngestGraphState, IngestGraphStepInfo
from langgraph.runtime import Runtime
from integrations.mineru import mineru_client


async def pdf_to_md(state: IngestGraphState, runtime: Runtime[IngestGraphStepInfo]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="pdf转md", status="running"))

    # 将pdf上传到minerU, 然后下载minerU转换后的压缩包url，获取md文件的存放路径
    try:
        full_zip_url = await mineru_client.upload_file(state.file_path)
        if not full_zip_url:
            writer(IngestGraphStepInfo(name="pdf转md", status="failed", error="文件上传Minio失败"))
            return {"should_continue": False, "error": "文件上传Minio失败"}
        md_file_path = await mineru_client.download_markdown(full_zip_url, state.markdown_dir)
        if not md_file_path:
            writer(IngestGraphStepInfo(name="pdf转md", status="failed", error="Minio解析文件失败"))
            return {"should_continue": False, "error": "Minio解析文件失败"}
        return {"markdown_file": md_file_path}
    except Exception as e:
        writer(IngestGraphStepInfo(name="pdf转md", status="failed", error=str(e)))
        return {"should_continue": False, "error": str(e)}
