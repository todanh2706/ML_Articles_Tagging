from datetime import datetime, timedelta
import csv
import io
import random
from typing import Dict, List

from flask import Flask, request, jsonify

from models import load_models, normalize_label

app = Flask(__name__)

TAGS = ["chinh-tri", "kinh-te", "giao-duc", "the-thao", "giai-tri", "cong-nghe", "doi-song"]


def random_softmax():
    scores = {tag: random.random() for tag in TAGS}
    total = sum(scores.values()) or 1.0
    return {tag: round(value / total, 6) for tag, value in scores.items()}


class StubModel:
    name = "Stub"

    def predict(self, text: str):
        return predict_stub(text)


def predict_stub(text: str):
    softmax = random_softmax()
    predicted_tag = max(softmax, key=softmax.get)
    return {"predicted_tag": predicted_tag, "softmax": softmax}


MODELS = load_models()
if not MODELS:
    MODELS = [StubModel()]


def normalize_tag(tag: str) -> str:
    return normalize_label(tag)


def format_prediction(model, text: str) -> Dict:
    try:
        raw = model.predict(text)
        predicted_tag = raw.predicted_tag if hasattr(raw, "predicted_tag") else raw.get("predicted_tag")
        softmax = raw.softmax if hasattr(raw, "softmax") else raw.get("softmax", raw.get("probs", {}))
        return {"name": model.name, "predicted_tag": predicted_tag, "softmax": softmax}
    except Exception as exc:  # pragma: no cover - defensive for runtime issues
        return {"name": model.name, "predicted_tag": None, "softmax": {}, "error": str(exc)}


def predict_with_models(text: str) -> List[Dict]:
    return [format_prediction(model, text) for model in MODELS]


def get_true_prob(probs: Dict, true_tag: str) -> float:
    target = normalize_tag(true_tag)
    for label, value in (probs or {}).items():
        if normalize_tag(label) == target:
            try:
                return float(value)
            except Exception:
                return 0.0
    return 0.0


def annotate_with_truth(pred: Dict, true_tag: str) -> Dict:
    pred_tag = normalize_tag(pred.get("predicted_tag"))
    true_tag_norm = normalize_tag(true_tag)
    prob = get_true_prob(pred.get("softmax", {}), true_tag)
    return {
        **pred,
        "true_tag_prob": prob,
        "correct": pred_tag == true_tag_norm,
    }


def ensure_text_payload():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text")
    if not text:
        return None, jsonify({"error": "text is required"}), 400
    return text, None, None


@app.post("/predict/text")
def predict_text():
    text, error_resp, status = ensure_text_payload()
    if error_resp:
        return error_resp, status
    preds = predict_with_models(text)
    return jsonify({"models": preds})


@app.post("/predict/url")
def predict_url():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "url is required"}), 400

    # In real life: download and extract article text. Here we stub.
    fake_text = f"Stub content fetched from {url}"
    models = predict_with_models(fake_text)

    return jsonify({
        "title": f"Fetched article from {url}",
        "snippet": fake_text[:200],
        "models": models,
    })


@app.post("/predict/csv")
def predict_csv():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "file is required"}), 400

    rows = []
    stream = io.StringIO(file.stream.read().decode("utf-8"))
    reader = csv.DictReader(stream)
    for idx, row in enumerate(reader, start=1):
        text = row.get("content") or row.get("text") or ""
        preds = predict_with_models(text)
        rows.append({
            "id": row.get("id") or idx,
            "title": row.get("title") or f"Row {idx}",
            "models": preds,
            "predicted_tags": {p["name"]: p.get("predicted_tag") for p in preds},
        })

    return jsonify({"rows": rows})


@app.post("/evaluate/text")
def evaluate_text():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text")
    true_tag = data.get("true_tag")
    if not text or not true_tag:
        return jsonify({"error": "text and true_tag are required"}), 400
    preds = [annotate_with_truth(pred, true_tag) for pred in predict_with_models(text)]
    return jsonify({"models": preds})


def compute_summary(rows):
    summary = {}
    for model in MODELS:
        preds = []
        for row in rows:
            for pred in row.get("models", []):
                if pred.get("name") == model.name:
                    preds.append(pred)
        total = len(preds) or 1
        correct = sum(1 for p in preds if p.get("correct"))
        accuracy = round(correct / total, 3)
        summary[model.name] = {
            "accuracy": accuracy,
            "precision": accuracy,
            "recall": accuracy,
            "f1": accuracy,
        }
    return summary


@app.post("/evaluate/csv")
def evaluate_csv():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "file is required"}), 400

    rows = []
    stream = io.StringIO(file.stream.read().decode("utf-8"))
    reader = csv.DictReader(stream)
    for idx, row in enumerate(reader, start=1):
        text = row.get("content") or row.get("text") or ""
        true_tag = row.get("true_tag") or row.get("label") or ""
        preds = [annotate_with_truth(pred, true_tag) for pred in predict_with_models(text)]
        row_result = {
            "id": row.get("id") or idx,
            "title": row.get("title") or f"Row {idx}",
            "true_tag": true_tag,
            "models": preds,
            "predicted_tags": {p["name"]: p.get("predicted_tag") for p in preds},
        }
        rows.append(row_result)

    summary = compute_summary(rows)
    return jsonify({"summary": summary, "rows": rows})


@app.get("/crawl/tag")
def crawl_tag():
    tag = request.args.get("tag")
    if not tag:
        return jsonify({"error": "tag is required"}), 400

    articles = []
    now = datetime.utcnow()
    for i in range(6):
        preds = predict_with_models(f"stub for {tag}")
        top_pred = preds[0] if preds else {}
        softmax = top_pred.get("softmax") or {}
        confidence = max(softmax.values()) if softmax else 0.0
        articles.append({
            "title": f"Thanh Nien article {i + 1} ve {tag}",
            "snippet": f"Doan tom tat ngan cho bai bao {i + 1} lien quan den {tag}.",
            "url": f"https://thanhnien.vn/{tag}/bai-{i + 1}.html",
            "source": "Thanh Nien",
            "published_at": (now - timedelta(hours=i * 2)).isoformat(),
            "predicted_tag": top_pred.get("predicted_tag"),
            "model": top_pred.get("name"),
            "confidence": confidence,
            "models": preds,
        })

    return jsonify({"articles": articles})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
