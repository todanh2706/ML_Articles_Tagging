export function formatModels(models) {
  if (!Array.isArray(models)) return [];
  return models.map((m) => {
    const softmax = m.softmax || {};
    const softmaxList = Object.entries(softmax)
      .map(([label, prob]) => ({ label, prob: Number(prob) }))
      .sort((a, b) => b.prob - a.prob);
    return {
      ...m,
      softmax,
      softmaxList,
    };
  });
}

export function topPrediction(models) {
  if (!Array.isArray(models) || models.length === 0) return '';

  const votes = new Map();
  models.forEach((model) => {
    const tag = model.predicted_tag;
    if (!tag) return;

    let topProb = 0;
    if (Array.isArray(model.softmaxList) && model.softmaxList.length > 0) {
      const candidate = Number(model.softmaxList[0].prob);
      topProb = Number.isFinite(candidate) ? candidate : 0;
    } else if (model.softmax && typeof model.softmax === 'object') {
      const values = Object.values(model.softmax)
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value));
      if (values.length > 0) {
        topProb = Math.max(...values);
      }
    }

    const entry = votes.get(tag) || { count: 0, bestProb: 0 };
    entry.count += 1;
    entry.bestProb = Math.max(entry.bestProb, topProb);
    votes.set(tag, entry);
  });

  if (votes.size === 0) return '';

  let bestTag = '';
  let bestCount = -1;
  let bestProb = -1;
  votes.forEach((value, tag) => {
    if (value.count > bestCount || (value.count === bestCount && value.bestProb > bestProb)) {
      bestTag = tag;
      bestCount = value.count;
      bestProb = value.bestProb;
    }
  });

  return bestTag;
}
