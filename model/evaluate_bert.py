import torch
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from pyvi import ViTokenizer
from tqdm import tqdm

# --- CẤU HÌNH ---
MODEL_PATH = "./final_model"
TEST_DATA_DIR = "./test"
MAX_LENGTH = 256
BATCH_SIZE = 32
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Thể thao", "Thế giới", "Giáo dục", "Kinh tế", 
    "Chính trị", "Sức khỏe", "Thời sự"
]

# --- DATASET CLASS ---
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- HÀM HỖ TRỢ ---
def load_data(data_dir):
    try:
        x_path = os.path.join(data_dir, "X_test_bert.csv")
        y_path = os.path.join(data_dir, "y_test_bert.csv")
        
        print(f"Loading test data from {data_dir}...")
        df_x = pd.read_csv(x_path)
        df_y = pd.read_csv(y_path)
        
        texts = df_x['content_segmented'].astype(str).tolist()
        labels = df_y['label_encoded'].tolist()
        
        return texts, labels
    except FileNotFoundError:
        print(f"Error: Không tìm thấy file dữ liệu test tại {data_dir}")
        sys.exit(1)

def plot_confusion_matrix(y_true, y_pred, classes, output_file="confusion_matrix_result.png"):
    """Vẽ Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Thực tế (True Label)', fontsize=12)
    plt.xlabel('Dự đoán (Predicted Label)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"--> Đã lưu biểu đồ Confusion Matrix tại: {output_file}")
    plt.close()

def plot_overall_metrics(acc, pre, rec, f1, output_file="overall_metrics_result.png"):
    """Vẽ biểu đồ cột cho 4 chỉ số tổng quát"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [acc, pre, rec, f1]
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3'] # Màu sắc đẹp mắt (Seaborn palette)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=colors, alpha=0.9, width=0.6)
    
    # Trang trí biểu đồ
    plt.title('Hiệu năng Tổng quát trên tập Test', fontsize=16, pad=20)
    plt.ylabel('Giá trị (0.0 - 1.0)', fontsize=12)
    plt.ylim(0.8, 1.05) # Giới hạn trục Y để nhìn rõ sự chênh lệch (vì model > 90%)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hiển thị con số trên đầu cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"--> Đã lưu biểu đồ Overall Metrics tại: {output_file}")
    plt.close()

# --- MAIN FUNCTION ---
def evaluate():
    print(f"Running on device: {DEVICE}")
    
    # 1. Load Model
    print("Loading model & tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        sys.exit(1)

    # 2. Load Data
    test_texts, test_labels = load_data(TEST_DATA_DIR)

    # 3. Tokenize & Predict
    print("Tokenizing data...")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_dataset = NewsDataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Starting evaluation...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Tính toán Metrics
    # Tính Accuracy
    acc = accuracy_score(all_labels, all_preds)
    
    # Tính Precision, Recall, F1 (Weighted Average)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("="*30)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-" * 30)
    
    print("Chi tiết từng lớp:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))
    
    # 5. VẼ BIỂU ĐỒ
    # Biểu đồ 1: Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES, output_file="confusion_matrix_result.png")
    
    # Biểu đồ 2: Overall Metrics (Yêu cầu mới của bạn)
    plot_overall_metrics(acc, precision, recall, f1, output_file="overall_metrics_result.png")

if __name__ == "__main__":
    evaluate()