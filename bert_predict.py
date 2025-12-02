import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyvi import ViTokenizer
import sys

# Cấu hình in ra terminal (UTF-8)
sys.stdout.reconfigure(encoding='utf-8')

# 1. Cấu hình thiết bị
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Running on: {device}")

# 2. Định nghĩa nhãn (Mapping phải KHỚP với lúc train)
id2label = {
    0: "Thể thao",
    1: "Thế giới",
    2: "Giáo dục",
    3: "Kinh tế",
    4: "Chính trị",
    5: "Sức khỏe",
    6: "Thời sự"
}

# 3. Tải model (Chỉ tải 1 lần)
model_path = "./final_bert_model" # Đảm bảo đường dẫn đúng
print(f"Loading model from {model_path}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval() # Chuyển sang chế độ đánh giá (tắt dropout)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def predict_news_ranking(text, top_k=None):
    """
    Trả về danh sách xác suất của tất cả các nhãn.
    top_k: Nếu truyền số (ví dụ 3), chỉ trả về top 3 nhãn cao nhất.
    """
    # Bước A: Tách từ
    text_segmented = ViTokenizer.tokenize(text)
    
    # Bước B: Mã hóa
    inputs = tokenizer(text_segmented, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Bước C: Dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Bước D: Tính xác suất (Softmax)
    # probabilities là tensor chứa 7 con số, ví dụ: [0.01, 0.9, 0.05, ...]
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Bước E: Tạo danh sách kết quả
    ranking = []
    for idx, score in enumerate(probabilities):
        ranking.append({
            "label": id2label[idx],
            "confidence": float(score) # Chuyển về float của Python
        })
    
    # Bước F: Sắp xếp giảm dần theo độ tin cậy
    ranking.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Trả về Top K nếu được yêu cầu, ngược lại trả về hết
    if top_k:
        return ranking[:top_k]
    return ranking

# --- KHỐI TEST ---
if __name__ == "__main__":
    # Một ví dụ hơi "lai" giữa Công nghệ và Kinh tế để xem model phân vân thế nào
    sample_text = """
    Cổ phiếu của các công ty công nghệ lớn đồng loạt giảm điểm sau khi 
    Cục Dự trữ Liên bang Mỹ công bố điều chỉnh lãi suất mới. 
    Các nhà đầu tư đang lo ngại về suy thoái kinh tế toàn cầu.
    """
    
    print("-" * 40)
    print(f"Input: {sample_text.strip()}")
    print("-" * 40)
    
    # Gọi hàm dự đoán
    results = predict_news_ranking(sample_text)
    
    print("KẾT QUẢ PHÂN LOẠI CHI TIẾT:")
    for item in results:
        # In định dạng đẹp: Label (thụt lề) --> %
        print(f" - {item['label']:<15}: {item['confidence']:.4f} ({item['confidence']*100:.2f}%)")