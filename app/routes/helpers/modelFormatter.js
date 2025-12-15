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
  return models[0].predicted_tag || '';
}
