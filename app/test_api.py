"""
Quick manual API checks against the Flask service.
Run from the repo root or the app/ folder:
  python app/test_api.py

The first call targets POST /evaluate/text (as requested).
"""

import json
import pathlib
import sys
from typing import Dict, Any

import requests


BASE_URL = "http://localhost:5001"


def call(method: str, path: str, **kwargs) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    resp = requests.request(method, url, timeout=15, **kwargs)
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    return {"status": resp.status_code, "body": body}


def test_evaluate_text():
    payload = {"text": "Bài viết thử nghiệm về kinh tế Việt Nam", "true_tag": "kinh-te"}
    return call("POST", "/evaluate/text", json=payload)


def test_predict_text():
    payload = {"text": "Đội tuyển Việt Nam thắng trận chung kết AFF Cup"}
    return call("POST", "/predict/text", json=payload)


def test_predict_url():
    payload = {"url": "https://example.com/demo-article"}
    return call("POST", "/predict/url", json=payload)


def test_predict_csv():
    csv_rows = "id,title,content,true_tag\n1,Demo,Chính trị và kinh tế Việt Nam,kinh-te\n"
    files = {"file": ("demo.csv", csv_rows, "text/csv")}
    return call("POST", "/evaluate/csv", files=files)


def main():
    tests = [
        ("evaluate_text", test_evaluate_text),
        ("predict_text", test_predict_text),
        ("predict_url", test_predict_url),
        ("evaluate_csv", test_predict_csv),
    ]

    for name, fn in tests:
        print(f"\n=== {name} ===")
        result = fn()
        print("Status:", result["status"])
        print("Body:", json.dumps(result["body"], ensure_ascii=False, indent=2) if isinstance(result["body"], (dict, list)) else result["body"])


if __name__ == "__main__":
    # Allow overriding BASE_URL via CLI arg: python app/test_api.py http://localhost:5001
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1].rstrip("/")
    main()
