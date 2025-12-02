# ============================================================
# File: train_fasttext.py
# Training FastText for News Topic Classification (Single Label)
# ============================================================

import os
import sqlite3
import unicodedata
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import fasttext

# ================================
# 1. CONFIG
# ================================
DB_PATH = "./dataset/articles.db"     # chỉnh path nếu cần
FT_DIR = "./fasttext_data"
os.makedirs(FT_DIR, exist_ok=True)

TRAIN_TXT = os.path.join(FT_DIR, "train.txt")
VALID_TXT = os.path.join(FT_DIR, "valid.txt")
TEST_TXT  = os.path.join(FT_DIR, "test.txt")
MODEL_PATH = os.path.join(FT_DIR, "news_main_tag.bin")


# ================================
# 2. HÀM TIỆN ÍCH
# ================================
def strip_accents(s: str) -> str:
    """
    Bỏ dấu tiếng Việt: 'Thể thao' -> 'The thao'
    """
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def normalize_label(label: str) -> str:
    """
    Chuẩn hoá nhãn về ASCII + gạch dưới, ví dụ:
    'Thể thao' -> '__label__the_thao'
    'Thời sự'  -> '__label__thoi_su'
    """
    if pd.isna(label):
        return ""
    label = str(label).strip().lower()
    label = strip_accents(label)                  # bỏ dấu
    label = re.sub(r"\s+", "_", label)            # space -> _
    label = re.sub(r"[^a-z0-9_]", "", label)      # chỉ giữ a-z, 0-9, _
    if not label:
        return ""
    return "__label__" + label


def clean_text(text: str) -> str:
    """
    Tiền xử lý đơn giản: lowercase, bỏ xuống dòng, strip.
    Có thể mở rộng filter HTML, ký tự đặc biệt nếu cần.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ")
    text = text.strip().lower()
    return text


def write_fasttext_file(path: str, series: pd.Series):
    """
    Ghi file cho FastText:
    Mỗi dòng: "__label__tag text..."
    Không dùng to_csv để tránh dấu ngoặc kép / escape ký tự.
    """
    with open(path, "w", encoding="utf-8") as f:
        for line in series:
            line = str(line).strip()
            if not line:
                continue
            f.write(line + "\n")


# ================================
# 3. ĐỌC DỮ LIỆU TỪ SQLITE
# ================================
print("🔍 Đang load dữ liệu từ SQLite...")

try:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    conn.close()
except Exception as e:
    print("❌ Lỗi đọc database:", e)
    raise e

print(f"✔ Số bài đọc được: {df.shape[0]}")


# ================================
# 4. TIỀN XỬ LÝ CHO MODELING
# ================================
print("🧹 Đang tiền xử lý dữ liệu...")

# Giữ các cột cần thiết
df_model = df[['title', 'content', 'main_tag', 'link']].copy()

# Loại bỏ dòng thiếu title/content/main_tag
df_model = df_model.dropna(subset=['title', 'content', 'main_tag'])

# Loại bỏ trùng link
df_model = df_model.drop_duplicates(subset=['link'])

print(f"✔ Còn lại sau lọc: {df_model.shape[0]} bài báo")

# Tạo text và label cho FastText
df_model['text'] = (df_model['title'] + " " + df_model['content']).apply(clean_text)
df_model['ft_label'] = df_model['main_tag'].apply(normalize_label)

# Bỏ các dòng không tạo được label hợp lệ
df_model = df_model[df_model['ft_label'] != ""].copy()

# Tạo dòng đúng format cho FastText
df_model['ft_line'] = df_model['ft_label'] + " " + df_model['text']

print("\n📌 Ví dụ 3 dòng FastText sau xử lý:")
for i in range(min(3, df_model.shape[0])):
    print(df_model['ft_line'].iloc[i][:200], "...")
print()

# ================================
# 5. CHIA TRAIN / VALID / TEST
# ================================
print("✂️ Đang chia train/valid/test...")

train_df, temp_df = train_test_split(
    df_model,
    test_size=0.30,
    random_state=42,
    stratify=df_model['ft_label']
)
valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df['ft_label']
)

print(f"✔ Train: {train_df.shape[0]}")
print(f"✔ Valid: {valid_df.shape[0]}")
print(f"✔ Test : {test_df.shape[0]}")

# Ghi file FastText
write_fasttext_file(TRAIN_TXT, train_df['ft_line'])
write_fasttext_file(VALID_TXT, valid_df['ft_line'])
write_fasttext_file(TEST_TXT,  test_df['ft_line'])

print(f"\n📁 Đã tạo file FastText tại folder: {FT_DIR}")
print(f"- {TRAIN_TXT}")
print(f"- {VALID_TXT}")
print(f"- {TEST_TXT}")

# ================================
# 6. TRAIN FASTTEXT
# ================================
# print("\n🚀 Bắt đầu train FastText...")

# model = fasttext.train_supervised(
#     input=TRAIN_TXT,
#     lr=0.3,
#     epoch=25,
#     dim=150,
#     wordNgrams=2,
#     loss="hs",   # single-label classification loss="softmax"
#     minn=2,
#     maxn=5,
#     verbose=2
# )

# model.save_model(MODEL_PATH)
# print(f"\n🎉 Đã lưu model: {MODEL_PATH}")

print("\n🚀 Bắt đầu train FastText bằng AUTO-TUNE...")

model = fasttext.train_supervised(
    input=TRAIN_TXT,
    autotuneValidationFile=VALID_TXT,
    autotuneDuration=600,     # 10 phút
    verbose=2
)

model.save_model(MODEL_PATH)
print(f"🎉 Đã lưu model autotuned tại: {MODEL_PATH}")

# ============================
# Quantization (optional)
# ============================
print("\n⚡ Đang quantize model...")

model_quant = fasttext.train_supervised(
    input=TRAIN_TXT,
    autotuneValidationFile=VALID_TXT,
    autotuneDuration=600,     # 10 phút
    verbose=2,
    autotuneModelSize="20M"
)

model_quant.save_model("fasttext_data/news_main_tag_quant.bin")
print("🎉 Đã lưu model quantized: news_main_tag_quant.bin")

# ================================
# 7. ĐÁNH GIÁ NHANH BẰNG FASTTEXT API
# ================================
print("\n📊 Đánh giá nhanh bằng FastText API...")

n_valid, p_at_1, r_at_1 = model.test(VALID_TXT)
print(f"VALID → Samples: {n_valid}, Precision@1: {p_at_1:.4f}, Recall@1: {r_at_1:.4f}")

n_test, p_test, r_test = model.test(TEST_TXT)
print(f"TEST  → Samples: {n_test}, Precision@1: {p_test:.4f}, Recall@1: {r_test:.4f}")

# ================================
# 8. ĐÁNH GIÁ CHI TIẾT (SKLEARN)
# ================================
print("\n🔬 Đánh giá chi tiết trên test set bằng sklearn...")

X_test = test_df['text'].tolist()
y_true = test_df['ft_label'].str.replace("__label__", "", regex=False).tolist()

y_pred = []
for txt in X_test:
    labels, probs = model.predict(txt, k=1)
    if len(labels) == 0:
        y_pred.append("unknown")
    else:
        pred = labels[0].replace("__label__", "")
        y_pred.append(pred)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, digits=4))

macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"⭐ Macro-F1 trên test set: {macro_f1:.4f}")

print("\n🎯 Training & Evaluation FastText hoàn tất!")
