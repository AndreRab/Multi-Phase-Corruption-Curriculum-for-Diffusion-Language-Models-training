import math
import warnings
from metrics.base_metric import BaseMetric
from metrics.metirc_enum import METRICS


class MauveMetric(BaseMetric):
    requires_text = True
    result_key = "mauve"

    def __init__(self, config):
        super().__init__(METRICS.MAUVE)
        self.config = config
        self.metadata = {}

    def compute(self, predictions, references):
        self.metadata = {"mauve_status": "requires_nonempty_texts_and_at_least_two_samples"}
        if len(predictions) < 2 or any(not text.strip() for text in predictions + references):
            return None
        import mauve
        if len(predictions) < 2000:
            warnings.warn("MAUVE is most reliable with a few thousand texts per corpus.")
        output = mauve.compute_mauve(
            p_text=references, q_text=predictions,
            featurize_model_name=self.config.mauve_model,
            max_text_length=self.config.mauve_max_text_length,
            device_id=self.config.mauve_device_id,
            seed=self.config.seed if self.config.seed is not None else 25,
            verbose=False,
        )
        value = float(output.mauve)
        self.metadata = {
            "mauve_status": "computed" if math.isfinite(value) else "nonfinite",
            "mauve_model": self.config.mauve_model,
            "mauve_max_text_length": self.config.mauve_max_text_length,
        }
        return value if math.isfinite(value) else None
