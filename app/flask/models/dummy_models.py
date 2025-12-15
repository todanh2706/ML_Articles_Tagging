"""
Placeholder module to show where real ML models would be loaded.
Replace the stub functions with actual model loading and inference.
"""


class DummyModel:
  def __init__(self, name):
    self.name = name

  def predict(self, text: str):
    # Replace with real inference.
    return {"label": "stub-tag", "probs": {"stub-tag": 1.0}}


def load_models():
  return [DummyModel("Model 1"), DummyModel("Model 2"), DummyModel("Model 3")]
