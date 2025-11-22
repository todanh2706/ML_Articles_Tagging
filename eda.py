import os
import re
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# ==========================
# 0. Cấu hình chung
# ==========================
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Folder lưu hình
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(name: str):
    """
    Lưu hình vào thư mục FIG_DIR với tên file name.png
    """
    filename = os.path.join(FIG_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved figure: {filename}")


# ==========================
# 1. Đọc dữ liệu từ SQLite
# ==========================
db_file_path = "./dataset/articles.db"  # nhớ đổi path nếu cần

try:
    conn = sqlite3.connect(db_file_path)
    print(f"Successfully connected to the database: {db_file_path}")

    sql_query = "SELECT * FROM articles"
    df = pd.read_sql_query(sql_query, conn)

    print("\nDataFrame head:")
    print(df.head())

except sqlite3.Error as e:
    print(f"Error connecting to or querying the db: {e}")
    raise e

finally:
    if 'conn' in locals() and conn:
        conn.close()
        print("Database connetion closed.")


# ==========================
# 2. Tiền xử lý chung cho EDA
# ==========================
df_eda = df.copy()

# Chuẩn hóa cột thời gian: "04/11/2025 12:43 GMT+7" -> datetime
df_eda['publication_datetime'] = pd.to_datetime(
    df_eda['publication_date'].astype(str).str.replace(' GMT+7', '', regex=False),
    format='%d/%m/%Y %H:%M',
    errors='coerce'
)

df_eda['pub_date'] = df_eda['publication_datetime'].dt.date
df_eda['pub_hour'] = df_eda['publication_datetime'].dt.hour

# Độ dài tiêu đề và nội dung
df_eda['title_len'] = df_eda['title'].astype(str).str.len()
df_eda['content_len'] = df_eda['content'].astype(str).str.len()


# ============================================================
# \subsection{Phân tích mối quan hệ giữa các thuộc tính}
# ============================================================

# 2.1. Quan hệ giữa source và main_tag
print("\n=== Bảng tần suất nguồn báo và tag chính ===")
cross_source_tag = pd.crosstab(df_eda['source'], df_eda['main_tag'])
print(cross_source_tag)

plt.figure()
sns.heatmap(cross_source_tag, annot=True, fmt='d')
plt.title("Phân bố bài viết theo Source và Main Tag")
plt.ylabel("Source")
plt.xlabel("Main Tag")
save_fig("rel_source_main_tag_heatmap")

# 2.2. Quan hệ giữa thời gian và số lượng bài viết (theo ngày)
print("\n=== Số lượng bài theo ngày ===")
articles_per_day = df_eda.groupby('pub_date')['link'].count()
print(articles_per_day.head())

plt.figure()
articles_per_day.plot(kind='line', marker='o')
plt.title("Số lượng bài viết theo ngày")
plt.xlabel("Ngày")
plt.ylabel("Số bài viết")
plt.xticks(rotation=45)
save_fig("rel_articles_per_day")

# 2.3. Số lượng bài theo giờ trong ngày
articles_per_hour = df_eda.groupby('pub_hour')['link'].count()

plt.figure()
articles_per_hour.plot(kind='bar')
plt.title("Số lượng bài viết theo giờ trong ngày")
plt.xlabel("Giờ")
plt.ylabel("Số bài viết")
save_fig("rel_articles_per_hour")

# 2.4. Quan hệ giữa main_tag và độ dài nội dung
print("\n=== Thống kê độ dài nội dung theo main_tag ===")
content_len_by_tag = df_eda.groupby('main_tag')['content_len'].agg(
    ['mean', 'median', 'min', 'max', 'count']
)
print(content_len_by_tag)

plt.figure()
sns.boxplot(data=df_eda, x='main_tag', y='content_len')
plt.title("Độ dài nội dung theo Main Tag")
plt.xlabel("Main Tag")
plt.ylabel("Độ dài nội dung (số ký tự)")
plt.xticks(rotation=45)
save_fig("rel_content_len_by_main_tag")

# 2.5. Quan hệ giữa source và độ dài tiêu đề
print("\n=== Thống kê độ dài tiêu đề theo source ===")
title_len_by_source = df_eda.groupby('source')['title_len'].agg(
    ['mean', 'median', 'min', 'max', 'count']
)
print(title_len_by_source)

plt.figure()
sns.boxplot(data=df_eda, x='source', y='title_len')
plt.title("Độ dài tiêu đề theo Source")
plt.xlabel("Source")
plt.ylabel("Độ dài tiêu đề (số ký tự)")
plt.xticks(rotation=45)
save_fig("rel_title_len_by_source")


# ===========================================
# \subsection{Kiểm tra chất lượng dữ liệu}
# ===========================================

# 3.1. Missing values
print("\n=== Kiểm tra giá trị thiếu ===")
missing_count = df_eda.isna().sum()
missing_ratio = df_eda.isna().mean()

quality_missing = pd.DataFrame({
    'missing_count': missing_count,
    'missing_ratio': missing_ratio
})
print(quality_missing)

# 3.2. Dòng trùng lặp
print("\n=== Kiểm tra trùng lặp ===")
num_duplicates = df_eda.duplicated().sum()
print(f"Số dòng trùng lặp hoàn toàn: {num_duplicates}")

num_duplicate_links = df_eda['link'].duplicated().sum()
print(f"Số link trùng lặp: {num_duplicate_links}")

if num_duplicate_links > 0:
    duplicate_links = df_eda[df_eda['link'].duplicated(keep=False)].sort_values('link')
    print("Các link bị trùng (nếu có):")
    print(duplicate_links[['link', 'title', 'publication_date']])

# 3.3. Kiểm tra tính hợp lệ của ngày giờ
print("\n=== Kiểm tra ngày giờ không parse được ===")
invalid_dates = df_eda[df_eda['publication_datetime'].isna()]
print(f"Số dòng có publication_date không parse được: {invalid_dates.shape[0]}")
print(invalid_dates[['publication_date', 'link']].head())

# 3.4. Phân bố độ dài title/content (phát hiện outlier)
print("\n=== Thống kê mô tả độ dài title và content ===")
print(df_eda[['title_len', 'content_len']].describe())

plt.figure()
sns.histplot(df_eda['title_len'], bins=30, kde=True)
plt.title("Phân bố độ dài tiêu đề")
plt.xlabel("Độ dài tiêu đề (số ký tự)")
save_fig("quality_title_len_distribution")

plt.figure()
sns.histplot(df_eda['content_len'], bins=30, kde=True)
plt.title("Phân bố độ dài nội dung")
plt.xlabel("Độ dài nội dung (số ký tự)")
save_fig("quality_content_len_distribution")


# ===========================================
# \subsection{EDA cho dữ liệu phi cấu trúc}
# ===========================================

# 4.1. Số câu ước lượng trong content
df_eda['num_sentences'] = df_eda['content'].astype(str).str.count(r'[.!?…]') + 1

print("\n=== Thống kê mô tả cho độ dài nội dung và số câu ===")
print(df_eda[['content_len', 'num_sentences']].describe())

plt.figure()
sns.scatterplot(data=df_eda, x='num_sentences', y='content_len')
plt.title("Quan hệ giữa số câu và độ dài nội dung")
plt.xlabel("Số câu (ước lượng)")
plt.ylabel("Độ dài nội dung (số ký tự)")
save_fig("text_rel_sentences_contentlen")


# 4.2. Tokenize đơn giản + đếm tần suất từ
def simple_tokenize(text):
    # Lowercase
    text = text.lower()
    # Loại bỏ ký tự đặc biệt (giữ lại chữ cái, số và khoảng trắng)
    text = re.sub(r'[^0-9a-zA-ZÀ-ỹ\s]', ' ', text)
    tokens = text.split()
    return tokens

all_tokens = []
for doc in df_eda['content'].dropna():
    all_tokens.extend(simple_tokenize(doc))

print(f"\nTổng số token (thô): {len(all_tokens)}")

counter = Counter(all_tokens)

# Stopwords cơ bản (có thể thay đổi cho phù hợp dữ liệu thực tế)
basic_stopwords = {
    'và', 'là', 'của', 'cho', 'với', 'một', 'những', 'các', 'được', 'trong',
    'đã', 'này', 'khi', 'tại', 'từ', 'đến', 'theo', 'ngày', 'năm', 'người',
    'ở', 'về', 'trên', 'đó', 'nhiều'
}

filtered_counter = Counter({
    token: freq for token, freq in counter.items()
    if token not in basic_stopwords and len(token) > 2
})

top_30_words = filtered_counter.most_common(30)
print("\n=== Top 30 từ (sau khi loại stopwords cơ bản) ===")
for w, f in top_30_words:
    print(f"{w}: {f}")

# 4.3. Vẽ biểu đồ top 30 từ
if len(top_30_words) > 0:
    top_words, top_freqs = zip(*top_30_words)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(top_words), y=list(top_freqs))
    plt.title("Top 30 từ xuất hiện nhiều nhất trong content\n(sau khi loại stopwords cơ bản)")
    plt.xlabel("Từ")
    plt.ylabel("Tần suất")
    plt.xticks(rotation=45, ha='right')
    save_fig("text_top_30_words")
else:
    print("Không đủ token để vẽ biểu đồ top từ.")

# 4.4. So sánh đặc trưng text giữa các main_tag / source
print("\n=== Độ dài trung bình nội dung theo main_tag ===")
content_len_tag = df_eda.groupby('main_tag')['content_len'].mean().sort_values(ascending=False)
print(content_len_tag)

plt.figure()
content_len_tag.plot(kind='bar')
plt.title("Độ dài trung bình nội dung theo Main Tag")
plt.xlabel("Main Tag")
plt.ylabel("Độ dài trung bình (số ký tự)")
plt.xticks(rotation=45)
save_fig("text_avg_content_len_by_main_tag")

print("\n=== Số câu trung bình theo source ===")
sentences_source = df_eda.groupby('source')['num_sentences'].mean().sort_values(ascending=False)
print(sentences_source)

plt.figure()
sentences_source.plot(kind='bar')
plt.title("Số câu trung bình trong content theo Source")
plt.xlabel("Source")
plt.ylabel("Số câu trung bình")
plt.xticks(rotation=45)
save_fig("text_avg_sentences_by_source")

print("\nEDA hoàn thành. Tất cả hình đã được lưu trong folder 'figures/'.")
