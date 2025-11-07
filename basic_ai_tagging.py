import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

TAGS = [
    "chính trị",
    "kinh tế",
    "thể thao",
    "giải trí",
    "giáo dục",
    "công nghệ",
    "đời sống",
    "pháp luật",
]

def tag_article(content: str) -> str:
    """
    Gửi nội dung bài báo lên model và nhận về tag (ở dạng text model trả về).
    Bạn có thể ép model trả JSON để dễ parse.
    """

    # để tránh quá dài / quá nhiều token, cắt bớt nếu cần
    content_short = content[:4000]

    instruction = f"""Bạn là hệ thống gắn nhãn chủ đề cho bài báo.

Hãy CHỈ chọn 1–3 tag phù hợp nhất trong danh sách sau:
{", ".join(TAGS)}

Yêu cầu:
- Chỉ trả về JSON đúng format:
  {{"tags": ["tag1", "tag2"]}}
- Không giải thích thêm, không ghi gì ngoài JSON.

Nội dung bài báo:
\"\"\"{content_short}\"\"\""""

    resp = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            # 2 header này tùy bạn có muốn dùng không
            "HTTP-Referer": "http://localhost",  
            "X-Title": "NewspaperTagger",
        },
        json={
            "model": "openrouter/polaris-alpha",
            "messages": [
                {"role": "user", "content": instruction}
            ]
        }
    )

    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    return text

import sqlite3
import pandas as pd


with sqlite3.connect('./dataset/articles.db') as conn:
    # Nếu có cột id thì dùng SELECT id, content
    # nếu không có id thì tạm SELECT rowid AS id, content
    df = pd.read_sql("SELECT rowid AS id, content FROM articles LIMIT 10", conn)

# nếu dataset lớn, bạn có thể LIMIT 10 để test trước:
# df = pd.read_sql("SELECT rowid AS id, content FROM articles LIMIT 10", conn)

for _, row in df.iterrows():
    article_id = row["id"]
    content = row["content"] or ""

    print(f"\n=== Article {article_id} ===")
    print(content[:200], "...\n")  # in 200 ký tự đầu
    # gọi model để gắn tag
    try:
        tag_response = tag_article(content)
        print("Model response:", tag_response)
    except Exception as e:
        print("Error tagging article:", e)
