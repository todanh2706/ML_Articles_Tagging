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

TAGS = ["the-thao", "the-gioi", "giao-duc", "kinh-te", "chinh-tri", "suc-khoe", "thoi-su"]

# Note: The mapping below expects normalized keys (lowercase, hyphens instead of underscores).
# The user provided tags have underscores and accents: "thể_thao", etc.
# normalize_label will convert "thể_thao" -> "thể-thao".
DISPLAY_NAMES = {
    # Accented keys (from new mapping)
    "thể-thao": "Thể thao",
    "thế-giới": "Thế giới",
    "giáo-dục": "Giáo dục",
    "kinh-tế": "Kinh tế",
    "chính-trị": "Chính trị",
    "sức-khỏe": "Sức khỏe",
    "thời-sự": "Thời sự",

    # Unaccented keys (legacy/FastText fallback)
    "the-thao": "Thể thao",
    "the-gioi": "Thế giới",
    "giao-duc": "Giáo dục",
    "kinh-te": "Kinh tế",
    "chinh-tri": "Chính trị",
    "suc-khoe": "Sức khỏe",
    "thoi-su": "Thời sự",
    "cong-nghe": "Công nghệ", # Legacy tag
    "doi-song": "Đời sống",   # Legacy tag
    "giai-tri": "Giải trí",   # Legacy tag
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
        predicted_tag = None
        softmax = {}

        if hasattr(raw, "predicted_tag"):
            predicted_tag = raw.predicted_tag
        elif isinstance(raw, dict):
            predicted_tag = raw.get("predicted_tag") or raw.get("label")

        if hasattr(raw, "softmax"):
            softmax = raw.softmax
        elif isinstance(raw, dict):
            softmax = raw.get("softmax") or raw.get("probs") or softmax
        elif isinstance(raw, tuple) and len(raw) == 2:
            labels, probs = raw
            softmax = {normalize_label(lab): float(prob) for lab, prob in zip(labels, probs)}

        if not predicted_tag and isinstance(softmax, dict) and softmax:
            predicted_tag = max(softmax, key=softmax.get, default=None)

        # Normalize predicted tag
        normalized_pred = normalize_tag(predicted_tag)
        
        # Convert softmax keys to display names
        display_softmax = {}
        for tag, score in softmax.items():
            norm_tag = normalize_tag(tag)
            display_name = DISPLAY_NAMES.get(norm_tag, norm_tag)
            display_softmax[display_name] = score
            
        display_predicted_tag = DISPLAY_NAMES.get(normalized_pred, normalized_pred)

        return {
            "name": model.name,
            "predicted_tag": display_predicted_tag,
            "softmax": display_softmax,
            "raw_tag": normalized_pred 
        }
    except Exception as exc:  # pragma: no cover - defensive for runtime issues
        return {"name": model.name, "predicted_tag": None, "softmax": {}, "error": str(exc)}


def predict_with_models(text: str) -> List[Dict]:
    return [format_prediction(model, text) for model in MODELS]


def iter_feed_articles(source: str, tag: str, max_items: int) -> Iterable[Dict]:
    source_map = TAG_RSS_MAP.get(source, {})
    rss_url = source_map.get(tag) or source_map.get("home")
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
        
        # Simpler: In format_prediction we added "raw_tag". Let's use that if available.
        raw_pred_tag = m.get("raw_tag")
        if not raw_pred_tag:
             raw_pred_tag = normalize_tag(pred_tag)

        # To find prob of 'tag' (slug), we look for its display name in probs
        display_name_of_tag = DISPLAY_NAMES.get(normalized_tag, normalized_tag)
        prob = probs.get(display_name_of_tag, 0.0)
        
        is_match = (raw_pred_tag == normalized_tag)
        
        if is_match and prob > 0:
            consensus_hits += 1
        if is_match and prob > best_prob:
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
        pairs = []
        for row in rows:
            true_tag = normalize_tag(row.get("true_tag") or "")
            pred_tag = ""
            for pred in row.get("models", []):
                if pred.get("name") == model.name:
                    pred_tag = normalize_tag(pred.get("predicted_tag") or "")
                    break
            if true_tag or pred_tag:
                pairs.append((true_tag, pred_tag))

        total = len(pairs)
        correct = sum(1 for true_tag, pred_tag in pairs if true_tag == pred_tag)
        accuracy = round(correct / total, 3) if total else 0.0

        labels = sorted({t for t, _ in pairs if t} | {p for _, p in pairs if p})
        precision = recall = f1 = 0.0
        if labels:
            precision_sum = 0.0
            recall_sum = 0.0
            f1_sum = 0.0
            for label in labels:
                tp = sum(1 for t, p in pairs if t == label and p == label)
                fp = sum(1 for t, p in pairs if t != label and p == label)
                fn = sum(1 for t, p in pairs if t == label and p != label)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1_score = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
                precision_sum += prec
                recall_sum += rec
                f1_sum += f1_score

            precision = round(precision_sum / len(labels), 3)
            recall = round(recall_sum / len(labels), 3)
            f1 = round(f1_sum / len(labels), 3)

        summary[model.name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
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

    limit = request.args.get("limit", default=30, type=int)
    min_conf = request.args.get("min_conf", default=0.5, type=float)
    requested_min_consensus = request.args.get("min_consensus", type=int)
    model_count = max(len(MODELS), 1)
    if requested_min_consensus is None:
        min_consensus = 2 if model_count >= 2 else 1
    else:
        min_consensus = max(0, min(requested_min_consensus, model_count))

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
