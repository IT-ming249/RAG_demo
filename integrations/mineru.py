import asyncio
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiofiles
import httpx

from conf import app_config
from core.log import logger


class MineruIntegration:
    def __init__(self):
        self.base_url = app_config.mineru.base_url
        self.token = app_config.mineru.token
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=30)

    async def upload_file(self, file_path: Path) -> str | None:
        url = f"{self.base_url}/file-urls/batch"
        payload = {
            "files": [{"name": file_path.name}],
            "model_version": "vlm",
        }
        signed_url_response = await self.client.post(url, json=payload)
        signed_url_response.raise_for_status()

        resp_data = signed_url_response.json()
        signed_url = resp_data["data"]["file_urls"][0]
        batch_id = resp_data["data"]["batch_id"]

        async with aiofiles.open(file_path, mode="rb") as fp:
            file_data = await fp.read()

        async with httpx.AsyncClient(timeout=60) as upload_client:
            upload_response = await upload_client.put(
                signed_url,
                content=file_data,
            )
        upload_response.raise_for_status()

        poll_url = f"{self.base_url}/extract-results/batch/{batch_id}"
        poll_max_times = 100
        poll_interval = 3
        polled_times = 0

        while True:
            await asyncio.sleep(poll_interval)

            poll_response = await self.client.get(poll_url, timeout=10)
            poll_response.raise_for_status()

            poll_data = poll_response.json()
            extract_result = poll_data["data"]["extract_result"]
            if extract_result:
                result_item = extract_result[0]
                state = result_item["state"]
                if state == "done":
                    full_zip_url = result_item.get("full_zip_url")
                    if full_zip_url:
                        return full_zip_url
                elif state == "failed":
                    error = result_item.get("err_msg", "MinerU 解析失败")
                    raise RuntimeError(f"{batch_id} 解析失败: {error}")
                else:
                    logger.debug(f"{batch_id} 轮询第 {polled_times + 1} 次，状态: {state}")

            polled_times += 1
            if polled_times >= poll_max_times:
                logger.info(f"{batch_id} 轮询超时")
                return None

    async def download_markdown(self, full_zip_url: str, output_dir: Path) -> str | None:
        pdf_stem = output_dir.name
        zip_file_name = Path(unquote(urlparse(full_zip_url).path)).name or f"{pdf_stem}_result.zip"

        logger.info(f"开始下载 MinerU 解析结果 ZIP 包: {full_zip_url}")
        response = await self.client.get(full_zip_url, timeout=120)
        response.raise_for_status()

        if output_dir.exists():
            await asyncio.to_thread(shutil.rmtree, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            prefix=f"{pdf_stem}_",
            suffix=f"_{zip_file_name}",
            delete=False,
        ) as temp_zip_file:
            zip_save_path = Path(temp_zip_file.name)

        try:
            async with aiofiles.open(zip_save_path, mode="wb") as fp:
                await fp.write(response.content)
            logger.info(f"MinerU 解析结果 ZIP 包下载完成: {zip_save_path}")

            def _extract_zip():
                output_dir_resolved = output_dir.resolve()
                with zipfile.ZipFile(zip_save_path, "r") as zip_file:
                    for member in zip_file.infolist():
                        member_path = (output_dir / member.filename).resolve()
                        if member_path != output_dir_resolved and output_dir_resolved not in member_path.parents:
                            raise RuntimeError(f"ZIP 包中包含非法路径: {member.filename}")
                    zip_file.extractall(output_dir)

            await asyncio.to_thread(_extract_zip)
        finally:
            if zip_save_path.exists():
                await asyncio.to_thread(zip_save_path.unlink)

        logger.info(f"MinerU 解析结果 ZIP 包解压完成: {output_dir}")

        md_file_list = list(output_dir.rglob("*.md"))
        if not md_file_list:
            raise FileNotFoundError(f"未找到 Markdown 文件: {output_dir}")

        target_md_file = None
        for md_file in md_file_list:
            if md_file.stem == pdf_stem:
                target_md_file = md_file
                break

        if not target_md_file:
            for md_file in md_file_list:
                if md_file.name.lower() == "full.md":
                    target_md_file = md_file
                    break

        if not target_md_file:
            target_md_file = md_file_list[0]

        if target_md_file.stem != pdf_stem:
            new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
            try:
                target_md_file.rename(new_md_path)
                target_md_file = new_md_path
            except OSError as exc:
                logger.warning(f"Markdown 文件重命名失败，继续使用原文件: {exc}")

        return str(target_md_file.resolve())


mineru_client = MineruIntegration()


if __name__ == "__main__":
    async def test():
        full_zip_url = await mineru_client.upload_file(Path("uploads/test.pdf"))
        logger.info(full_zip_url)
        md_file_path = await mineru_client.download_markdown(full_zip_url, Path("uploads/"))  # type: ignore[arg-type]
        logger.info(md_file_path)

    asyncio.run(test())
