import pandas as pd
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt # Thư viện vẽ biểu đồ
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# --- CẤU HÌNH HỆ THỐNG ---
# 1. Cấu hình in ra terminal (UTF-8) để tránh lỗi font trên Windows
sys.stdout.reconfigure(encoding='utf-8')

# 2. Cấu hình thiết bị (Device)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: CUDA")
else:
    device = torch.device("cpu")
    print("Device: CPU")

# --- SIÊU THAM SỐ (HYPERPARAMETERS) ---
MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
NUM_LABELS = 7 

# --- HÀM HỖ TRỢ ---
def load_data(data_dir, type_data):
    try:
        x_path = os.path.join(data_dir, f"X_{type_data}_bert.csv")
        y_path = os.path.join(data_dir, f"y_{type_data}_bert.csv")
        
        print(f"Loading {type_data} data...")
        df_x = pd.read_csv(x_path)
        df_y = pd.read_csv(y_path)
        
        # Đảm bảo dữ liệu đầu vào là string
        texts = df_x['content_segmented'].astype(str).tolist()
        labels = df_y['label_encoded'].tolist()
        
        return texts, labels
    except FileNotFoundError:
        print(f"Error: Không tìm thấy file dữ liệu cho {type_data} tại {data_dir}")
        sys.exit(1)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    # Dùng weighted vì dữ liệu có thể lệch nhẹ, hoặc macro tùy yêu cầu báo cáo
    f1 = f1_score(labels, preds, average='weighted')
    
    return {'accuracy': acc, 'f1': f1}

# --- HÀM VẼ BIỂU ĐỒ ---
def plot_training_history(log_history, output_dir='./results'):
    """
    Hàm vẽ biểu đồ gộp:
    - Bên trái: Training Loss (theo steps) và Validation Loss (theo epoch).
    - Bên phải: Validation Accuracy & F1 (theo epoch).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Tách dữ liệu
    train_loss = []
    train_steps = []
    
    val_loss = []
    val_acc = []
    val_f1 = []
    val_epochs = []

    for entry in log_history:
        # Lấy dữ liệu Train (ghi nhận theo step)
        if 'loss' in entry and 'step' in entry:
            train_loss.append(entry['loss'])
            train_steps.append(entry['step'])
        
        # Lấy dữ liệu Validation (ghi nhận theo epoch)
        if 'eval_loss' in entry:
            val_loss.append(entry['eval_loss'])
            val_acc.append(entry['eval_accuracy'])
            val_f1.append(entry['eval_f1'])
            val_epochs.append(entry['epoch'])

    # 2. Tạo khung hình (1 hàng, 2 cột)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- BIỂU ĐỒ 1: LOSS (QUAN TRỌNG ĐỂ SOI OVERFITTING) ---
    # Vẽ Train Loss (dạng đường mờ vì nó dao động mạnh)
    ax1.plot(train_steps, train_loss, label='Training Loss (Steps)', alpha=0.3, color='gray')
    
    # Làm mượt Train Loss (Moving Average) để dễ nhìn hơn
    if len(train_loss) > 10:
        window = 10
        train_loss_smooth = pd.Series(train_loss).rolling(window=window).mean()
        ax1.plot(train_steps, train_loss_smooth, label='Train Loss (Smoothed)', color='#1f77b4', linewidth=2)
    
    # Vẽ Val Loss (dạng điểm nối đường) - Cần map epoch sang step tương ứng để vẽ cùng trục
    # Ước lượng step tại mỗi epoch: max_step * (epoch / max_epoch)
    # Tuy nhiên để đơn giản, ta dùng trục phụ hoặc chỉ vẽ các điểm mốc.
    # Cách tốt nhất: Vẽ Val Loss theo trục Epoch phía trên hoặc chỉ so sánh giá trị.
    # Ở đây mình vẽ Val Loss dưới dạng đường ngang qua các epoch (marker to)
    
    # (Để đơn giản hóa trục x, mình sẽ vẽ Val Loss lên cùng nhưng lưu ý trục x là Steps)
    # Ta lấy bước step cuối cùng của mỗi epoch để chấm điểm Val Loss
    steps_per_epoch = train_steps[-1] / val_epochs[-1] if val_epochs else 0
    val_steps = [e * steps_per_epoch for e in val_epochs]
    
    ax1.plot(val_steps, val_loss, label='Validation Loss', marker='o', color='red', linewidth=2, linestyle='--')
    
    ax1.set_title('Training vs Validation Loss', fontsize=14)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- BIỂU ĐỒ 2: METRICS (ACCURACY & F1) ---
    ax2.plot(val_epochs, val_acc, label='Validation Accuracy', marker='s', color='green', linewidth=2)
    ax2.plot(val_epochs, val_f1, label='Validation F1-Score', marker='^', color='purple', linewidth=2, linestyle='--')
    
    # Note: HuggingFace Trainer mặc định không tính Train Accuracy.
    # Nếu muốn vẽ Train Accuracy, cần phải Custom Trainer rất phức tạp.
    # Thay vào đó, Train Loss giảm sâu (ở biểu đồ 1) chính là dấu hiệu Train Accuracy tăng cao.
    
    ax2.set_title('Validation Metrics over Epochs', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score (0.0 - 1.0)')
    ax2.set_ylim(0.8, 1.0) # Zoom vào khoảng cao để nhìn rõ
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Thêm chú thích cho các điểm giá trị
    for i, txt in enumerate(val_acc):
        ax2.annotate(f"{txt:.4f}", (val_epochs[i], val_acc[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # 3. Lưu ảnh
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_combined_charts.png')
    plt.savefig(output_path)
    print(f"--> Đã lưu biểu đồ gộp tại: {output_path}")
    plt.close()
# --- MAIN ---
if __name__ == "__main__":
    print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN ---")

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Load Data
    # Lưu ý: Cần đảm bảo folder 'train' và 'val' chứa file csv đúng định dạng
    train_texts, train_labels = load_data('./train', 'train')
    val_texts, val_labels = load_data('./val', 'val')

    print("Tokenizing data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    # 3. Load Model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,          # Ghi log loss sau mỗi 50 steps
        eval_strategy="epoch",     # Đánh giá sau mỗi epoch (để vẽ biểu đồ cho đẹp)
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Optimization for Mac M series
        dataloader_pin_memory=False if device.type == 'mps' else True,
        use_mps_device=True if device.type == 'mps' else False
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Start Training
    print("Training...")
    trainer.train()

    # 7. VẼ BIỂU ĐỒ (Bước mới thêm)
    print("Đang vẽ biểu đồ quá trình học...")
    plot_training_history(trainer.state.log_history)

    # 8. Save Final Model
    print("Saving final model to './final_model'...")
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    
    print("\n--- BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST ---")
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # 1. Load dữ liệu Test
    # Đảm bảo bạn có file X_test_bert.csv và y_test_bert.csv trong thư mục ./test
    try:
        test_texts, test_labels = load_data('./test', 'test')
        print("Tokenizing test data...")
        test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        test_dataset = NewsDataset(test_encodings, test_labels)
        
        # 2. Dự đoán trên tập Test bằng Trainer
        print("Running prediction on Test set...")
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # 3. Vẽ Confusion Matrix (Yêu cầu )
        print("Generating Confusion Matrix...")
        cm = confusion_matrix(true_labels, preds)
        plt.figure(figsize=(10, 8))
        target_names = ["Thể thao", "Thế giới", "Giáo dục", "Kinh tế", "Chính trị", "Sức khỏe", "Thời sự"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix on Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('./results/confusion_matrix.png')
        print(f"--> Đã lưu Confusion Matrix tại: ./results/confusion_matrix.png")
        plt.close()
        
        # 4. Lưu các trường hợp dự đoán sai (Error Analysis - Yêu cầu )
        print("Saving error cases for analysis...")
        error_indices = np.where(preds != true_labels)[0]
        
        error_data = []
        for idx in error_indices:
            error_data.append({
                'text': test_texts[idx],
                'true_label': target_names[true_labels[idx]],
                'predicted_label': target_names[preds[idx]]
            })
            
        df_errors = pd.DataFrame(error_data)
        df_errors.to_csv('./results/error_analysis.csv', index=False, encoding='utf-8')
        print(f"--> Đã lưu {len(df_errors)} trường hợp dự đoán sai tại: ./results/error_analysis.csv")
        print("Bạn hãy dùng file này để viết phần 'Phân tích lỗi' trong báo cáo.")
        
    except Exception as e:
        print(f"Không thể chạy đánh giá trên tập Test: {e}")
        print("Lưu ý: Hãy đảm bảo bạn đã có folder 'test' chứa file csv giống cấu trúc train/val.")