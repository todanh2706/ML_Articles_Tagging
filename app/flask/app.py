from datetime import datetime, timedelta, timezone
import csv
import io
import random
import time
from typing import Dict, List, Optional, Iterable, Tuple

from flask import Flask, request, jsonify
from newspaper import Article, Config
import feedparser

from models import load_models, normalize_label

app = Flask(__name__)

TAGS = ["chinh-tri", "kinh-te", "giao-duc", "the-thao", "giai-tri", "cong-nghe", "doi-song"]
TAG_RSS_MAP = {
    "vnexpress": {
        "home": "https://vnexpress.net/rss/tin-moi-nhat.rss",
    },
    "thanhnien": {
        "home": "https://thanhnien.vn/rss/home.rss",
    },
    "laodong": {
        "home": "https://laodong.vn/rss/home.rss",
    },
}


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


def fetch_article(url: str) -> Optional[Dict[str, str]]:
    """
    Try to extract article content/title via newspaper3k.
    Returns None on failure to let caller fall back to stub.
    """
    cfg = Config()
    cfg.browser_user_agent = "Mozilla/5.0 (compatible; ML-Articles-Tagger/1.0)"
    cfg.request_timeout = 10
    cfg.memoize_articles = False
    article = Article(url, language="vi", config=cfg)
    article.download()
    article.parse()
    text = (article.text or "").replace("\n", " ").strip()
    if not text:
        return None
    title = (article.title or "").strip() or f"Fetched from {url}"
    snippet = text[:200]
    return {"title": title, "text": text, "snippet": snippet}


def parse_published(entry) -> datetime:
    ts = None
    if getattr(entry, "published_parsed", None):
        ts = entry.published_parsed
    elif getattr(entry, "updated_parsed", None):
        ts = entry.updated_parsed
    if ts:
        return datetime.fromtimestamp(time.mktime(ts), tz=timezone.utc)
    return datetime.now(tz=timezone.utc)


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


def iter_feed_articles(source: str, tag: str, max_items: int) -> Iterable[Dict]:
    rss_url = TAG_RSS_MAP.get(source, {}).get("home")
    if not rss_url:
        return []

    feed = feedparser.parse(rss_url)
    if getattr(feed, "bozo", False):
        return []

    for entry in feed.entries[:max_items]:
        url = getattr(entry, "link", None)
        if not url:
            continue
        yield {
            "source": source,
            "url": url,
            "title": (getattr(entry, "title", "") or "").strip(),
            "summary": (getattr(entry, "summary", "") or "").strip(),
            "published_at": parse_published(entry),
        }


def score_article_for_tag(text: str, tag: str) -> Tuple[List[Dict], str, str, float, int]:
    models = predict_with_models(text)
    normalized_tag = normalize_tag(tag)

    best_model = ""
    best_pred_tag = ""
    best_prob = 0.0
    consensus_hits = 0

    for m in models:
        probs = m.get("softmax") or {}
        pred_tag = m.get("predicted_tag") or ""
        prob = get_true_prob(probs, normalized_tag)
        if normalize_tag(pred_tag) == normalized_tag and prob > 0:
            consensus_hits += 1
        if prob > best_prob:
            best_prob = prob
            best_model = m.get("name") or ""
            best_pred_tag = pred_tag

    return models, best_model, best_pred_tag, best_prob, consensus_hits


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

    title = f"Fetched article from {url}"
    snippet = f"Stub content fetched from {url}"
    text = snippet

    try:
        article = fetch_article(url)
        if article:
            title = article["title"]
            snippet = article["snippet"]
            text = article["text"]
    except Exception as exc:  # pragma: no cover - defensive only
        snippet = f"Fallback stub due to error: {exc}"

    models = predict_with_models(text)

    return jsonify({
        "title": title,
        "snippet": snippet,
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

    normalized_tag = normalize_tag(tag)

    sources_param = request.args.get("sources")
    if sources_param:
        requested_sources = [s.strip() for s in sources_param.split(",") if s.strip()]
    else:
        requested_sources = list(TAG_RSS_MAP.keys())

    limit = request.args.get("limit", type=int) or 30
    min_conf = request.args.get("min_conf", type=float) or 0.5
    min_consensus = request.args.get("min_consensus", type=int) or 2

    per_source_max = max(5, limit // max(len(requested_sources), 1) * 2)

    collected = []

    for source in requested_sources:
        for raw_article in iter_feed_articles(source, normalized_tag, per_source_max):
            try:
                full = fetch_article(raw_article["url"])
            except Exception:
                full = None

            if full and full.get("text"):
                text = full["text"]
                snippet = full["snippet"]
                title = full["title"]
            else:
                text = raw_article["summary"] or raw_article["title"]
                snippet = (raw_article["summary"] or text)[:200]
                title = raw_article["title"]

            models, best_model, best_pred_tag, best_prob, consensus_hits = score_article_for_tag(
                text, normalized_tag
            )

            if best_prob < min_conf:
                continue
            if consensus_hits < min_consensus:
                continue

            collected.append({
                "title": title,
                "snippet": snippet,
                "url": raw_article["url"],
                "source": source,
                "published_at": raw_article["published_at"].isoformat(),
                "predicted_tag": best_pred_tag,
                "model": best_model,
                "confidence": round(float(best_prob), 3),
                "consensus_hits": consensus_hits,
                "match": normalize_tag(best_pred_tag or "") == normalized_tag,
                "models": models,
            })

    collected.sort(key=lambda a: a["published_at"], reverse=True)
    collected = collected[:limit]

    return jsonify({
        "tag": tag,
        "normalized_tag": normalized_tag,
        "sources": requested_sources,
        "min_conf": min_conf,
        "min_consensus": min_consensus,
        "articles": collected,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
