"""测试 DeepSeek API Key 能否正常调用文本 / 视觉模型。

用法（任选其一）:
    python test_api.py                          # 从 conf/app_config.yaml 自动读取
    python test_api.py --api-key sk-xxx         # 显式指定 Key

环境变量:
    DEEPSEEK_API_KEY
"""
import argparse
import json
import os
import urllib.error
import urllib.request

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL = "deepseek-chat"
VLM_MODEL = "deepseek-v4-flash-vision-exp"


def load_key_from_config():
    """从 conf/app_config.yaml 读取 llm 段的 api_key，无需任何第三方库。

    优先用 omegaconf（项目环境已安装时），失败则回退到极简行解析。
    """
    try:
        from conf import app_config
        return app_config.llm.api_key
    except Exception:
        pass

    # 极简解析：app_config.yaml 顶层段（0 缩进）下的 api_key 子键（缩进）
    current_section = None
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf", "app_config.yaml")
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


def call_chat(url, api_key, model, timeout=60):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "你好，请只回复：OK"}],
        "max_tokens": 10,
    }
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
    return data["choices"][0]["message"]["content"], data.get("model")


def test(name, url, api_key, model):
    if not api_key:
        print(f"[{name}] 跳过：未提供 API Key")
        return
    print(f"[{name}] 测试 {model} @ {url} ...")
    try:
        content, resp_model = call_chat(url, api_key, model)
        print(f"[{name}] SUCCESS 模型回复: {content!r}  (响应 model: {resp_model})")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[{name}] FAIL  HTTP {e.code}: {body}")
    except Exception as e:
        print(f"[{name}] FAIL  {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="测试 DeepSeek API Key（文本 + 视觉模型）")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    args = parser.parse_args()

    api_key = args.api_key or load_key_from_config()
    if not api_key:
        print("未找到 API Key。请用 --api-key 参数提供，")
        print("或确认 conf/app_config.yaml 中 llm 段的 api_key 已填写。")
        return

    test("LLM(文本)", DEEPSEEK_URL, api_key, LLM_MODEL)
    test("VLM(视觉)", DEEPSEEK_URL, api_key, VLM_MODEL)


if __name__ == "__main__":
    main()
