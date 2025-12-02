import pandas as pd
import torch
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# Set encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: CUDA")
else:
    device = torch.device("cpu")
    print("Device: CPU")

# Hyperparameters
MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
NUM_LABELS = 7 

def load_data(data_dir, type_data):
    try:
        x_path = os.path.join(data_dir, f"X_{type_data}_bert.csv")
        y_path = os.path.join(data_dir, f"y_{type_data}_bert.csv")
        
        print(f"Loading {type_data} data...")
        df_x = pd.read_csv(x_path)
        df_y = pd.read_csv(y_path)
        
        # Use 'content_segmented' column as input feature
        texts = df_x['content_segmented'].astype(str).tolist()
        labels = df_y['label_encoded'].tolist()
        
        return texts, labels
    except FileNotFoundError:
        print(f"Error: Data files for {type_data} not found.")
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
    f1 = f1_score(labels, preds, average='weighted')
    
    return {'accuracy': acc, 'f1': f1}

if __name__ == "__main__":
    print("--- Starting Training Process ---")

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Load and Process Data
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
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Optimization for Mac M2 (MPS)
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

    # 7. Save Final Model
    print("Saving final model to './final_model'...")
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    
    print("Done.")