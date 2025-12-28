import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# Utils
# =========================
def load_split(x_path, y_path):
    X = pd.read_csv(x_path, encoding="utf-8-sig")
    y = pd.read_csv(y_path, encoding="utf-8-sig")
    return X, y


def drop_nan_labels(texts: pd.Series, labels: pd.Series):
    mask = labels.notna()
    return texts[mask], labels[mask]


def structured_multiclass_hinge_loss(scores: np.ndarray, y_true: np.ndarray) -> float:
    """
    Multi-class structured hinge loss (Crammer-Singer style):
        L = mean( max(0, 1 - s_y + max_{j!=y} s_j ) )
    scores: (N, C) decision_function outputs
    y_true: (N,) int class ids mapped to indices [0..C-1]
    """
    N, C = scores.shape
    correct = scores[np.arange(N), y_true]
    tmp = scores.copy()
    tmp[np.arange(N), y_true] = -np.inf
    max_other = np.max(tmp, axis=1)
    loss = np.maximum(0.0, 1.0 - correct + max_other)
    return float(np.mean(loss))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


# =========================
# Main
# =========================
def main():
    # ---- paths ----
    ROOT = Path(__file__).resolve().parent      # .../model
    BASE = ROOT.parent                           # parent chứa train/val/test và model/

    X_train, y_train = load_split(BASE / "train/X_train_basic.csv", BASE / "train/y_train_basic.csv")
    X_val,   y_val   = load_split(BASE / "val/X_val_basic.csv",   BASE / "val/y_val_basic.csv")
    X_test,  y_test  = load_split(BASE / "test/X_test_basic.csv",  BASE / "test/y_test_basic.csv")

    TEXT_COL = "content_final"
    LABEL_COL = "label_encoded"

    train_texts = X_train[TEXT_COL].astype(str)
    val_texts   = X_val[TEXT_COL].astype(str)
    test_texts  = X_test[TEXT_COL].astype(str)

    train_labels = y_train[LABEL_COL]
    val_labels   = y_val[LABEL_COL]
    test_labels  = y_test[LABEL_COL]

    train_texts, train_labels = drop_nan_labels(train_texts, train_labels)
    val_texts,   val_labels   = drop_nan_labels(val_texts, val_labels)
    test_texts,  test_labels  = drop_nan_labels(test_texts, test_labels)

    # ---- load trained pipeline (LinearSVC pipeline) ----
    artifacts_dir = ensure_dir(ROOT / "artifacts")
    model_path = artifacts_dir / "tfidf_svm.joblib"
    labels_path = artifacts_dir / "tfidf_svm_labels.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không thấy model đã train tại: {model_path}\n"
            f"Hãy chạy script train trước để tạo artifacts/tfidf_svm.joblib."
        )

    best_model = joblib.load(model_path)

    label_names = None
    if labels_path.exists():
        label_info = json.loads(labels_path.read_text(encoding="utf-8"))
        label_names = label_info.get("label_names", None)

    # =========================
    # (A) EVALUATE: VALIDATION + TEST
    # =========================
    val_pred = best_model.predict(val_texts)
    val_acc = accuracy_score(val_labels, val_pred)
    val_f1m = f1_score(val_labels, val_pred, average="macro")

    test_pred = best_model.predict(test_texts)
    test_acc = accuracy_score(test_labels, test_pred)
    test_f1m = f1_score(test_labels, test_pred, average="macro")

    pr, rc, f1, _ = precision_recall_fscore_support(
        test_labels, test_pred, average="macro", zero_division=0
    )

    print("=== VALIDATION ===")
    print(f"Accuracy: {val_acc:.6f}")
    print(f"F1-macro: {val_f1m:.6f}")
    print(classification_report(val_labels, val_pred))

    print("\n=== TEST ===")
    print(f"Accuracy: {test_acc:.6f}")
    print(f"Precision(macro): {pr:.6f}")
    print(f"Recall(macro):    {rc:.6f}")
    print(f"F1-macro:         {test_f1m:.6f}")
    print(classification_report(test_labels, test_pred))

    # Save text reports
    reports_dir = ensure_dir(artifacts_dir / "reports")
    (reports_dir / "val_classification_report.txt").write_text(
        classification_report(val_labels, val_pred), encoding="utf-8"
    )
    (reports_dir / "test_classification_report.txt").write_text(
        classification_report(test_labels, test_pred), encoding="utf-8"
    )

    # =========================
    # (A.1) OVERALL METRICS BAR CHART (TEST)
    # =========================
    plots_dir = ensure_dir(artifacts_dir / "plots")

    overall_fig = plt.figure(figsize=(8, 6))
    ax = overall_fig.add_subplot(111)

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metrics_vals = [float(test_acc), float(pr), float(rc), float(f1)]

    bars = ax.bar(metrics_names, metrics_vals)
    ax.set_title("Hiệu năng Tổng quát trên tập Test", fontweight="bold")
    ax.set_ylabel("Giá trị (0.0 - 1.0)")
    ax.set_ylim(0.8, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for b, v in zip(bars, metrics_vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.005,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    overall_fig.tight_layout()
    overall_path = plots_dir / "overall_metrics_test.png"
    overall_fig.savefig(overall_path, dpi=200)
    plt.close(overall_fig)

    # =========================
    # (A.2) CONFUSION MATRIX (TEST)
    # =========================
    cm = confusion_matrix(test_labels, test_pred)
    classes = best_model.classes_

    if label_names:
        tick_labels = [label_names.get(str(c), str(c)) for c in classes]
    else:
        tick_labels = [str(c) for c in classes]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tick_labels)
    disp.plot(ax=ax, cmap=None, values_format="d", xticks_rotation=45)
    ax.set_title("Confusion Matrix (Test) - TF-IDF + SVM")
    fig.tight_layout()
    cm_path = plots_dir / "confusion_matrix_test.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    # =========================
    # (B) LEARNING CURVES theo epoch (Train/Val Loss + Accuracy)
    # =========================
    # NOTE: LinearSVC không có epoch/loss per-epoch -> dùng SGDClassifier(hinge) để mô phỏng Linear SVM.
    tfidf: TfidfVectorizer = best_model.named_steps["tfidf"]

    Xtr = tfidf.fit_transform(train_texts)
    Xva = tfidf.transform(val_texts)

    classes_sorted = np.array(sorted(pd.unique(train_labels)))
    class_to_index = {c: i for i, c in enumerate(classes_sorted)}

    ytr = np.array([class_to_index[c] for c in train_labels.values])
    yva = np.array([class_to_index.get(c, -1) for c in val_labels.values])
    valid_mask_va = yva >= 0
    Xva2 = Xva[valid_mask_va]
    yva2 = yva[valid_mask_va]

    clf = SGDClassifier(
        loss="hinge",
        alpha=1e-5,
        learning_rate="optimal",
        random_state=42,
        tol=None,
        max_iter=1,
    )

    EPOCHS = 20
    BATCH_SIZE = 512

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    clf.partial_fit(Xtr[:1], ytr[:1], classes=np.arange(len(classes_sorted)))

    rng = np.random.default_rng(42)
    n = Xtr.shape[0]

    for epoch in range(1, EPOCHS + 1):
        idx = np.arange(n)
        rng.shuffle(idx)

        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            clf.partial_fit(Xtr[batch_idx], ytr[batch_idx])

        tr_scores = clf.decision_function(Xtr)
        va_scores = clf.decision_function(Xva2)

        if tr_scores.ndim == 1:
            tr_scores = np.vstack([-tr_scores, tr_scores]).T
        if va_scores.ndim == 1:
            va_scores = np.vstack([-va_scores, va_scores]).T

        tr_loss = structured_multiclass_hinge_loss(tr_scores, ytr)
        va_loss = structured_multiclass_hinge_loss(va_scores, yva2)

        tr_pred = clf.predict(Xtr)
        va_pred = clf.predict(Xva2)

        tr_acc = accuracy_score(ytr, tr_pred)
        va_acc = accuracy_score(yva2, va_pred)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss={tr_loss:.4f}, Train Acc={tr_acc:.4f} | "
            f"Val Loss={va_loss:.4f}, Val Acc={va_acc:.4f}"
        )

    epochs = np.arange(1, EPOCHS + 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hinge Loss (structured)")
    ax.set_title("Learning Curve - Loss (TF-IDF + Linear SVM via SGD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = plots_dir / "learning_curve_loss.png"
    fig.savefig(loss_path, dpi=200)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_accs, label="Train Accuracy")
    ax.plot(epochs, val_accs, label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve - Accuracy (TF-IDF + Linear SVM via SGD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = plots_dir / "learning_curve_accuracy.png"
    fig.savefig(acc_path, dpi=200)
    plt.close(fig)

    # =========================
    # Save metrics summary JSON
    # =========================
    summary = {
        "validation": {"accuracy": float(val_acc), "f1_macro": float(val_f1m)},
        "test": {
            "accuracy": float(test_acc),
            "precision_macro": float(pr),
            "recall_macro": float(rc),
            "f1_macro": float(test_f1m),
        },
        "learning_curves": {
            "epochs": int(EPOCHS),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accs,
            "val_acc": val_accs,
        },
        "plots": {
            "overall_metrics_test": str(overall_path),
            "confusion_matrix_test": str(cm_path),
            "learning_curve_loss": str(loss_path),
            "learning_curve_accuracy": str(acc_path),
        },
    }
    (reports_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSaved outputs:")
    print(f"- {overall_path}")
    print(f"- {cm_path}")
    print(f"- {loss_path}")
    print(f"- {acc_path}")
    print(f"- {reports_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
