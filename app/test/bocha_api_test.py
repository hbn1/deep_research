import json
import os
import urllib.error
import urllib.request


BASE_URL = "https://api.bocha.cn/v1"
WEB_SEARCH_ENDPOINT = f"{BASE_URL}/web-search"
AI_SEARCH_ENDPOINT = f"{BASE_URL}/ai-search"

WEB_SEARCH_PAYLOAD = {
    "query": "OpenClaw latest usage trends",
    "summary": True,
    "freshness": "noLimit",
    "count": 10,
}

AI_SEARCH_PAYLOAD = {
    "query": "Beijing weather",
    "freshness": "noLimit",
    "count": 10,
    "answer": False,
    "stream": False,
}


def call_bocha_api(url: str, api_key: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response_text = response.read().decode("utf-8")
    return json.loads(response_text)


def summarize_result(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    print("top_level_keys:", list(result.keys()))
    if "code" in result:
        print("code:", result.get("code"))
    if "msg" in result:
        print("msg:", result.get("msg"))
    data = result.get("data")
    if isinstance(data, dict):
        print("data_keys:", list(data.keys()))
        web_pages = data.get("webPages")
        if isinstance(web_pages, list):
            print("webPages_count:", len(web_pages))
            if web_pages and isinstance(web_pages[0], dict):
                first = web_pages[0]
                preview = {
                    "name": first.get("name"),
                    "url": first.get("url"),
                    "summary": first.get("summary"),
                }
                print("webPages_first:", json.dumps(preview, ensure_ascii=False))
    print("raw:", json.dumps(result, ensure_ascii=False)[:1200])


def main() -> None:
    api_key = os.getenv("BOCHA_API_KEY", "").strip()
    if not api_key:
        print("BOCHA_API_KEY is not set; skipping live Bocha API check.")
        return

    for name, endpoint, payload in (
        ("Web Search", WEB_SEARCH_ENDPOINT, WEB_SEARCH_PAYLOAD),
        ("AI Search", AI_SEARCH_ENDPOINT, AI_SEARCH_PAYLOAD),
    ):
        try:
            result = call_bocha_api(endpoint, api_key, payload)
            summarize_result(name, result)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            print(f"\n=== {name} HTTPError ===")
            print("status:", exc.code)
            print("body:", body[:1200])
        except Exception as exc:
            print(f"\n=== {name} Exception ===")
            print(str(exc))


if __name__ == "__main__":
    main()
