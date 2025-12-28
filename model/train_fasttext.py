"""
Train fastText for Vietnamese news classification and produce
- learning curves (loss, accuracy) on train/validation
- test metrics (accuracy, precision, recall, F1)
- confusion matrix plot
"""

import os
import sqlite3
import unicodedata
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import fasttext

# ------------------
# CONFIG
# ------------------
DB_PATH = "./dataset/articles.db"
FT_DIR = "./fasttext_data"
FIG_DIR = "./figures"
os.makedirs(FT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TRAIN_TXT = os.path.join(FT_DIR, "train.txt")
VALID_TXT = os.path.join(FT_DIR, "valid.txt")
TEST_TXT = os.path.join(FT_DIR, "test.txt")
MODEL_BEST_PATH = os.path.join(FT_DIR, "news_main_tag_best.bin")

# Training sweep to get learning curves
EPOCH_GRID = [5, 10, 15, 20, 25]
FT_HPARAMS = dict(
    lr=0.3,
    dim=150,
    wordNgrams=2,
    loss="softmax",  # single-label
    minn=2,
    maxn=5,
    verbose=2,
)


# ------------------
# HELPERS
# ------------------
def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def normalize_label(label: str) -> str:
    if pd.isna(label):
        return ""
    label = str(label).strip().lower()
    label = strip_accents(label)
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^a-z0-9_]", "", label)
    return "__label__" + label if label else ""


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\n", " ").strip().lower()
    return text


def write_fasttext_file(path: str, series: pd.Series) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in series:
            line = str(line).strip()
            if line:
                f.write(line + "\n")


def predict_with_probs(model, texts: List[str]) -> Tuple[List[str], List[List[float]], List[str]]:
    """Return top-1 labels, probability vectors aligned with label_order."""
    label_order = [lab.replace("__label__", "") for lab in model.get_labels()]
    k = len(label_order)
    top_labels = []
    prob_vectors = []
    for txt in texts:
        labs, probs = model.predict(txt, k=k)
        prob_map = {labs[i].replace("__label__", ""): float(probs[i]) for i in range(len(labs))}
        vector = [prob_map.get(lab, 0.0) for lab in label_order]
        prob_vectors.append(vector)
        top_labels.append(labs[0].replace("__label__", "") if labs else "")
    return top_labels, prob_vectors, label_order


def eval_split(model, texts: List[str], true_labels: List[str]) -> Dict[str, float]:
    preds, prob_vectors, label_order = predict_with_probs(model, texts)
    acc = accuracy_score(true_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro", zero_division=0
    )
    label_to_idx = {lab: i for i, lab in enumerate(label_order)}
    eps = 1e-12
    logloss_terms = []
    for y, probs in zip(true_labels, prob_vectors):
        idx = label_to_idx.get(y, None)
        if idx is None:
            continue
        p = max(min(probs[idx], 1 - eps), eps)
        logloss_terms.append(-np.log(p))
    logloss = float(np.mean(logloss_terms)) if logloss_terms else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, logloss=logloss)


def plot_learning_curves(history: List[Dict[str, float]], out_path: str) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_logloss"] for h in history]
    val_loss = [h["val_logloss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Learning Curve - Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, marker="o", label="Train Acc")
    plt.plot(epochs, val_acc, marker="s", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_conf_mat(model, texts: List[str], labels: List[str], label_order: List[str], out_path: str) -> None:
    preds, _, _ = predict_with_probs(model, texts)
    cm = confusion_matrix(labels, preds, labels=label_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix - fastText")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ------------------
# MAIN
# ------------------
if __name__ == "__main__":
    print("Loading data from SQLite...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    conn.close()

    df_model = df[["title", "content", "main_tag", "link"]].dropna(subset=["title", "content", "main_tag"]).copy()
    df_model = df_model.drop_duplicates(subset=["link"])
    df_model["text"] = (df_model["title"] + " " + df_model["content"]).apply(clean_text)
    df_model["ft_label"] = df_model["main_tag"].apply(normalize_label)
    df_model = df_model[df_model["ft_label"] != ""].copy()
    df_model["ft_line"] = df_model["ft_label"] + " " + df_model["text"]

    print(f"Remaining samples after cleaning: {df_model.shape[0]}")

    train_df, temp_df = train_test_split(
        df_model, test_size=0.30, random_state=42, stratify=df_model["ft_label"]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["ft_label"]
    )

    print(f"Train: {train_df.shape[0]}, Valid: {valid_df.shape[0]}, Test: {test_df.shape[0]}")

    write_fasttext_file(TRAIN_TXT, train_df["ft_line"])
    write_fasttext_file(VALID_TXT, valid_df["ft_line"])
    write_fasttext_file(TEST_TXT, test_df["ft_line"])

    # Prepare raw texts/labels for metrics
    train_texts = train_df["text"].tolist()
    valid_texts = valid_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    train_labels = train_df["ft_label"].str.replace("__label__", "", regex=False).tolist()
    valid_labels = valid_df["ft_label"].str.replace("__label__", "", regex=False).tolist()
    test_labels = test_df["ft_label"].str.replace("__label__", "", regex=False).tolist()

    best_f1 = -1.0
    best_model = None
    best_label_order: List[str] = []
    history: List[Dict[str, float]] = []

    print("\nTraining fastText across epochs for learning curves...")
    for ep in EPOCH_GRID:
        print(f"\n--- Training with epoch={ep} ---")
        model = fasttext.train_supervised(
            input=TRAIN_TXT,
            epoch=ep,
            **FT_HPARAMS,
        )

        train_metrics = eval_split(model, train_texts, train_labels)
        val_metrics = eval_split(model, valid_texts, valid_labels)

        history.append(
            dict(
                epoch=ep,
                train_acc=train_metrics["acc"],
                train_logloss=train_metrics["logloss"],
                val_acc=val_metrics["acc"],
                val_logloss=val_metrics["logloss"],
                val_f1=val_metrics["f1"],
            )
        )

        print(
            f"[epoch={ep}] Train Acc={train_metrics['acc']:.4f} | "
            f"Val Acc={val_metrics['acc']:.4f}, Val F1={val_metrics['f1']:.4f}, Val LogLoss={val_metrics['logloss']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_model = model
            best_label_order = [lab.replace("__label__", "") for lab in model.get_labels()]

    if best_model is None:
        raise RuntimeError("No model was trained.")

    best_model.save_model(MODEL_BEST_PATH)
    print(f"\nSaved best model (Val F1={best_f1:.4f}) to {MODEL_BEST_PATH}")

    # Learning curves
    lc_path = os.path.join(FIG_DIR, "fasttext_learning_curves.png")
    plot_learning_curves(history, lc_path)
    print(f"Saved learning curves to {lc_path}")

    # Test evaluation with best model
    print("\nEvaluating best model on TEST...")
    preds, _, _ = predict_with_probs(best_model, test_texts)
    print("\n=== Classification report (TEST) ===")
    print(classification_report(test_labels, preds, digits=4))

    cm_path = os.path.join(FIG_DIR, "fasttext_confusion_matrix.png")
    plot_conf_mat(best_model, test_texts, test_labels, best_label_order, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Quick aggregate metrics
    acc = accuracy_score(test_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(test_labels, preds, average="macro", zero_division=0)
    print(
        f"TEST metrics -> Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}"
    )

