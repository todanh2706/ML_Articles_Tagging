import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover - joblib ships with scikit-learn, keep graceful
    joblib = None


def project_root() -> Path:
    """
    Resolve a workable project root.
    Tries to climb upward until it sees common markers; falls back to the Flask app folder.
    This avoids IndexError when the code is packaged inside a shallow Docker image (/app).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if any((parent / marker).exists() for marker in ("fasttext_data", "dataset", "model", ".git")):
            return parent
    return here.parent


def load_label_mapping() -> List[str]:
    """
    Optional manual label mapping shared across models.
    Default path can be overridden via LABEL_MAPPING_PATH.
    """
    root = project_root()
    default_path = root / "models" / "label_mapping.json"
    path = Path(os.getenv("LABEL_MAPPING_PATH", default_path))
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [normalize_label(x) for x in data]
        if isinstance(data, dict):
            labels = data.get("labels") or data.get("label_names") or data.get("mapping")
            if isinstance(labels, list):
                return [normalize_label(x) for x in labels]
    except Exception:
        return []
    return []


def normalize_label(label: str) -> str:
    """Lowercase, drop fastText prefixes, and unify separators for comparison."""
    if label is None:
        return ""
    return str(label).replace("__label__", "").replace("_", "-").strip().lower()


def softmax_scores(scores: List[float], labels: List[str]) -> Dict[str, float]:
    """Convert raw scores to a normalized probability dictionary."""
    if not scores:
        return {}
    max_score = max(scores)
    exps = [float(np.exp(s - max_score)) for s in scores]
    denom = sum(exps) or 1.0
    return {labels[i]: round(exps[i] / denom, 6) for i in range(len(labels))}


@dataclass
class Prediction:
    predicted_tag: str
    softmax: Dict[str, float]


class FastTextWrapper:
    def __init__(self, model_path: Path, name: str = "FastText", top_k: int = 3, label_map: Optional[List[str]] = None):
        import fasttext

        self.name = name
        self.model_path = Path(model_path)
        self.model = fasttext.load_model(str(self.model_path))
        labels = [normalize_label(label) for label in self.model.get_labels()]
        if label_map and len(label_map) == len(labels):
            labels = label_map
        self.labels = labels
        self.top_k = top_k

    def predict(self, text: str) -> Prediction:
        labels, probs = self.model.predict(text, k=min(self.top_k, len(self.labels)))
        cleaned = {normalize_label(lab): float(prob) for lab, prob in zip(labels, probs)}
        tag = max(cleaned, key=cleaned.get, default="")
        return Prediction(predicted_tag=tag, softmax=cleaned)


class PhoBERTWrapper:
    def __init__(self, model_dir: Path, name: str = "PhoBERT", max_length: int = 256, label_map: Optional[List[str]] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        id2label = {}
        if hasattr(self.model.config, "id2label"):
            id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        if label_map and len(label_map) == self.model.config.num_labels:
            self.labels = label_map
        elif id2label:
            self.labels = [normalize_label(id2label[i]) for i in sorted(id2label.keys())]
        else:
            self.labels = [f"label-{i}" for i in range(self.model.config.num_labels)]

    def predict(self, text: str) -> Prediction:
        import torch

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        softmax = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        tag = max(softmax, key=softmax.get, default="")
        return Prediction(predicted_tag=tag, softmax=softmax)


class TfidfSVMWrapper:
    def __init__(
        self,
        model_path: Path,
        labels_path: Optional[Path] = None,
        name: str = "TFIDF+LinearSVM",
        label_map: Optional[List[str]] = None,
    ):
        if joblib is None:
            raise ImportError("joblib is required to load the TF-IDF model.")

        self.name = name
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)
        self.label_names = self._load_label_metadata(labels_path)
        classes = list(self.pipeline.classes_)
        if label_map and len(label_map) == len(classes):
            self.labels = label_map
            self.label_names = {str(cls): label_map[idx] for idx, cls in enumerate(classes)}
        else:
            self.labels = [self.label_names.get(str(cls), str(cls)) for cls in classes]

    def _load_label_metadata(self, labels_path: Optional[Path]) -> Dict[str, str]:
        if labels_path and Path(labels_path).exists():
            try:
                data = json.loads(Path(labels_path).read_text(encoding="utf-8"))
                return data.get("label_names", {})
            except Exception:
                return {}
        return {}

    def predict(self, text: str) -> Prediction:
        scores = self.pipeline.decision_function([text])
        raw_scores = scores[0]

        # Handle binary case where LinearSVC returns a single score
        if np.isscalar(raw_scores):
            raw_scores = np.array([-raw_scores, raw_scores])
            classes = list(self.pipeline.classes_)
            if len(classes) == 1:
                classes = [classes[0], classes[0]]
        else:
            classes = list(self.pipeline.classes_)

        labels = [self.label_names.get(str(cls), str(cls)) for cls in classes]
        probs = softmax_scores(raw_scores.tolist(), labels)
        tag = max(probs, key=probs.get, default="")
        return Prediction(predicted_tag=tag, softmax=probs)


def load_models() -> List:
    """Load available models (fastText, PhoBERT, TF-IDF) if their artifacts exist."""
    models = []
    root = project_root()
    label_map = load_label_mapping()

    fasttext_path = Path(os.getenv("FASTTEXT_MODEL_PATH", root / "fasttext_data" / "news_main_tag.bin"))
    if fasttext_path.exists():
        try:
            models.append(FastTextWrapper(fasttext_path, name="FastText", label_map=label_map or None))
            print(f"[model] Loaded fastText from {fasttext_path}")
        except Exception as exc:
            print(f"[model] Skipped fastText ({exc})")
    else:
        print(f"[model] fastText artifact not found at {fasttext_path}, skipping.")

    phobert_dir = Path(os.getenv("PHOBERT_MODEL_DIR", root / "final_model"))
    if phobert_dir.exists():
        try:
            models.append(PhoBERTWrapper(phobert_dir, name="PhoBERT", label_map=label_map or None))
            print(f"[model] Loaded PhoBERT from {phobert_dir}")
        except Exception as exc:
            print(f"[model] Skipped PhoBERT ({exc})")
    else:
        print(f"[model] PhoBERT artifact not found at {phobert_dir}, skipping.")

    tfidf_path = Path(os.getenv("TFIDF_MODEL_PATH", root / "model" / "artifacts" / "tfidf_svm.joblib"))
    labels_path = Path(os.getenv("TFIDF_LABELS_PATH", root / "model" / "artifacts" / "tfidf_svm_labels.json"))
    if tfidf_path.exists():
        try:
            models.append(TfidfSVMWrapper(tfidf_path, labels_path, name="TFIDF+SVM", label_map=label_map or None))
            print(f"[model] Loaded TF-IDF+SVM from {tfidf_path}")
        except Exception as exc:
            print(f"[model] Skipped TF-IDF+SVM ({exc})")
    else:
        print(f"[model] TF-IDF+SVM artifact not found at {tfidf_path}, skipping.")

    return models


__all__ = [
    "load_models",
    "normalize_label",
    "Prediction",
    "FastTextWrapper",
    "PhoBERTWrapper",
    "TfidfSVMWrapper",
]
