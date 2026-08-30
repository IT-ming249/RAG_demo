"""测试 DeepSeek API Key 能否正常调用文本 / 视觉模型。

用法:
    python test_api.py                                    # 从 conf/app_config.yaml 自动读取
    python test_api.py --api-key sk-xxx                   # 显式指定 Key
    python test_api.py --image C:\\path\\pic.jpg          # 用本地图片测试视觉模型
    python test_api.py --image-url https://.../pic.jpg    # 用图片 URL 测试视觉模型

环境变量:
    DEEPSEEK_API_KEY
"""
import argparse
import base64
import json
import os
import urllib.error
import urllib.request

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL = "deepseek-chat"
VLM_MODEL = "deepseek-v4-flash-vision-exp"
# 默认示例图（阿里云域名，国内可访问；DeepSeek 服务器能正常下载）
DEFAULT_IMAGE_URL = "https://dashscope.aliyuncs.com/images/dog_and_girl.jpeg"


def load_key_from_config():
    """从 conf/app_config.yaml 读取 llm 段的 api_key，无需任何第三方库。

    优先用 omegaconf（项目环境已安装时），失败则回退到极简行解析。
    """
    try:
        from conf import app_config
        return app_config.llm.api_key
    except Exception:
        pass

    # 极简解析：项目根目录 conf/app_config.yaml 顶层段（0 缩进）下的 api_key 子键（缩进）
    current_section = None
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conf", "app_config.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if not line.startswith(" "):  # 0 缩进 = 顶层段名
                    current_section = stripped.rstrip(":")
                elif current_section == "llm" and stripped.startswith("api_key:"):
                    return stripped.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def image_to_data_uri(path):
    """本地图片转 base64 data URI。"""
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(ext, ext)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def call_chat(url, api_key, model, image=None, timeout=60):
    text = "这张图片里有什么？请用一句话回答。"
    if image:
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image}},
        ]
    else:
        content = "你好，请只回复：OK"
    payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 500}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0], data.get("model")


def test(name, url, api_key, model, image=None):
    if not api_key:
        print(f"[{name}] 跳过：未提供 API Key")
        return
    if image:
        print(f"[{name}] 测试 {model}（图片输入） @ {url} ...")
    else:
        print(f"[{name}] 测试 {model}（纯文本） @ {url} ...")
    try:
        choice, resp_model = call_chat(url, api_key, model, image)
        msg = choice.get("message", {})
        content = msg.get("content")
        reasoning = msg.get("reasoning_content")
        print(f"[{name}] SUCCESS  (响应 model: {resp_model})")
        print(f"[{name}]   content          : {content!r}")
        if reasoning:
            print(f"[{name}]   reasoning_content: {reasoning!r}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[{name}] FAIL  HTTP {e.code}: {body}")
    except Exception as e:
        print(f"[{name}] FAIL  {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="测试 DeepSeek API Key（文本 + 视觉模型）")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--image", help="本地图片路径，用于测试视觉模型")
    parser.add_argument("--image-url", help="图片 URL，用于测试视觉模型")
    args = parser.parse_args()

    api_key = args.api_key or load_key_from_config()
    if not api_key:
        print("未找到 API Key。请用 --api-key 参数提供，")
        print("或确认 conf/app_config.yaml 中 llm 段的 api_key 已填写。")
        return

    image = None
    if args.image:
        image = image_to_data_uri(args.image)
    elif args.image_url:
        image = args.image_url
    else:
        image = DEFAULT_IMAGE_URL  # 默认用国内可访问的示例图

    test("LLM(文本)", DEEPSEEK_URL, api_key, LLM_MODEL)
    test("VLM(视觉)", DEEPSEEK_URL, api_key, VLM_MODEL, image)


if __name__ == "__main__":
    main()
