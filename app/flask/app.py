from datetime import datetime, timedelta
import csv
import io
import random
from flask import Flask, request, jsonify

app = Flask(__name__)

TAGS = ["chinh-tri", "kinh-te", "giao-duc", "the-thao", "giai-tri", "cong-nghe", "doi-song"]
MODEL_NAMES = ["Model 1", "Model 2", "Model 3"]


def random_softmax():
    scores = {tag: random.random() for tag in TAGS}
    total = sum(scores.values()) or 1.0
    return {tag: round(value / total, 3) for tag, value in scores.items()}


def predict_stub(text: str):
    softmax = random_softmax()
    predicted_tag = max(softmax, key=softmax.get)
    return {"predicted_tag": predicted_tag, "softmax": softmax}


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

    models = []
    for name in MODEL_NAMES:
        pred = predict_stub(text)
        models.append({"name": name, **pred})
    return jsonify({"models": models})


@app.post("/predict/url")
def predict_url():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "url is required"}), 400

    # In real life: download and extract article text. Here we stub.
    fake_text = f"Stub content fetched from {url}"
    models = []
    for name in MODEL_NAMES:
        pred = predict_stub(fake_text)
        models.append({"name": name, **pred})

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
        preds = [predict_stub(row.get("content") or row.get("text") or "") for _ in MODEL_NAMES]
        rows.append({
            "id": row.get("id") or idx,
            "title": row.get("title") or f"Row {idx}",
            "model1_tag": preds[0]["predicted_tag"],
            "model2_tag": preds[1]["predicted_tag"],
            "model3_tag": preds[2]["predicted_tag"],
        })

    return jsonify({"rows": rows})


@app.post("/evaluate/text")
def evaluate_text():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text")
    true_tag = data.get("true_tag")
    if not text or not true_tag:
        return jsonify({"error": "text and true_tag are required"}), 400

    models = []
    for name in MODEL_NAMES:
        pred = predict_stub(text)
        true_prob = pred["softmax"].get(true_tag, 0)
        models.append({
            "name": name,
            "predicted_tag": pred["predicted_tag"],
            "true_tag_prob": true_prob,
            "correct": pred["predicted_tag"] == true_tag,
        })
    return jsonify({"models": models})


def compute_summary(rows, key):
    total = len(rows) or 1
    correct = sum(1 for r in rows if r.get(f"{key}_correct"))
    accuracy = round(correct / total, 3)
    # Placeholder: using accuracy for precision/recall/F1 in stub mode.
    return {
        "accuracy": accuracy,
        "precision": accuracy,
        "recall": accuracy,
        "f1": accuracy,
    }


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
        preds = [predict_stub(text) for _ in MODEL_NAMES]
        row_result = {
            "id": row.get("id") or idx,
            "title": row.get("title") or f"Row {idx}",
            "true_tag": true_tag,
            "model1_tag": preds[0]["predicted_tag"],
            "model2_tag": preds[1]["predicted_tag"],
            "model3_tag": preds[2]["predicted_tag"],
            "model1_correct": preds[0]["predicted_tag"] == true_tag,
            "model2_correct": preds[1]["predicted_tag"] == true_tag,
            "model3_correct": preds[2]["predicted_tag"] == true_tag,
        }
        rows.append(row_result)

    summary = {
        "model1": compute_summary(rows, "model1"),
        "model2": compute_summary(rows, "model2"),
        "model3": compute_summary(rows, "model3"),
    }
    return jsonify({"summary": summary, "rows": rows})


@app.get("/crawl/tag")
def crawl_tag():
    tag = request.args.get("tag")
    if not tag:
        return jsonify({"error": "tag is required"}), 400

    articles = []
    now = datetime.utcnow()
    for i in range(6):
        pred = predict_stub(f"stub for {tag}")
        articles.append({
            "title": f"Thanh Nien article {i + 1} ve {tag}",
            "snippet": f"Doan tom tat ngan cho bai bao {i + 1} lien quan den {tag}.",
            "url": f"https://thanhnien.vn/{tag}/bai-{i + 1}.html",
            "source": "Thanh Nien",
            "published_at": (now - timedelta(hours=i * 2)).isoformat(),
            "predicted_tag": pred["predicted_tag"],
            "confidence": max(pred["softmax"].values()),
        })

    return jsonify({"articles": articles})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
