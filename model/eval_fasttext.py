# ============================================================
# File: eval_fasttext_models.py
# Load 2 model (base + quant) và đánh giá trên VALID & TEST
# ============================================================

import os
import fasttext
from sklearn.metrics import classification_report, f1_score


# ================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ================================
FT_DIR = "./fasttext_data"

TRAIN_TXT = os.path.join(FT_DIR, "train.txt")   
VALID_TXT = os.path.join(FT_DIR, "valid.txt")
TEST_TXT  = os.path.join(FT_DIR, "test.txt")

BASE_MODEL_PATH  = os.path.join(FT_DIR, "news_main_tag.bin")
QUANT_MODEL_PATH = os.path.join(FT_DIR, "news_main_tag_quant.bin")


# ================================
# 2. HÀM ĐỌC valid/test TỪ FILE .TXT
# ================================
def load_fasttext_file(path: str):
    """
    Đọc file FastText: mỗi dòng dạng
        __label__xxx text ....
    Trả về:
        texts: list[str]
        labels: list[str] (không còn prefix __label__)
    """
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            raw_label, text = parts
            label = raw_label.replace("__label__", "")
            labels.append(label)
            texts.append(text)
    return texts, labels


# ================================
# 3. HÀM ĐÁNH GIÁ MỘT MÔ HÌNH
# ================================
def evaluate_model(name: str, model, X_valid, y_valid, X_test, y_test):
    print("\n===================================")
    print(f"🔍 ĐÁNH GIÁ MÔ HÌNH: {name}")
    print("===================================")

    # 3.1. Dùng API test() của FastText (validation)
    print("\n[FastText test() trên VALID]")
    n_valid, p_at_1_v, r_at_1_v = model.test(VALID_TXT)
    print(f"VALID → Samples: {n_valid}, Precision@1: {p_at_1_v:.4f}, Recall@1: {r_at_1_v:.4f}")

    # 3.2. Dùng API test() của FastText (test)
    print("\n[FastText test() trên TEST]")
    n_test, p_at_1_t, r_at_1_t = model.test(TEST_TXT)
    print(f"TEST  → Samples: {n_test}, Precision@1: {p_at_1_t:.4f}, Recall@1: {r_at_1_t:.4f}")

    # 3.3. Đánh giá chi tiết hơn bằng sklearn trên TEST
    print("\n[Sklearn Evaluation trên TEST]")

    y_pred = []
    for txt in X_test:
        labels, probs = model.predict(txt, k=1)
        if not labels:
            y_pred.append("unknown")
        else:
            pred = labels[0].replace("__label__", "")
            y_pred.append(pred)

    print("\n=== Classification Report (TEST) ===")
    print(classification_report(y_test, y_pred, digits=4))

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"⭐ Macro-F1 (TEST) của {name}: {macro_f1:.4f}")


# ================================
# 4. MAIN
# ================================
if __name__ == "__main__":
    # 4.1. Đọc VALID & TEST từ file
    print("📂 Đang load VALID & TEST từ file FastText...")
    X_valid, y_valid = load_fasttext_file(VALID_TXT)
    X_test,  y_test  = load_fasttext_file(TEST_TXT)

    print(f"✔ VALID samples: {len(X_valid)}")
    print(f"✔ TEST  samples: {len(X_test)}")

    # 4.2. Load model thường
    print("\n📦 Đang load BASE model...")
    base_model = fasttext.load_model(BASE_MODEL_PATH)
    print(f"✔ Đã load: {BASE_MODEL_PATH}")

    # 4.3. Nếu có model quant thì load
    quant_model = None
    if os.path.exists(QUANT_MODEL_PATH):
        print("\n📦 Đang load QUANTIZED model...")
        quant_model = fasttext.load_model(QUANT_MODEL_PATH)
        print(f"✔ Đã load: {QUANT_MODEL_PATH}")
    else:
        print(f"\n⚠ Không tìm thấy file quantized model: {QUANT_MODEL_PATH}")
        print("   (Bỏ qua phần đánh giá quantized model)")

    # 4.4. Đánh giá model thường
    evaluate_model("Base FastText", base_model, X_valid, y_valid, X_test, y_test)

    # 4.5. Đánh giá model quantized (nếu có)
    if quant_model is not None:
        evaluate_model("Quantized FastText", quant_model, X_valid, y_valid, X_test, y_test)

    print("\n🎯 Đã đánh giá xong tất cả các model.")
