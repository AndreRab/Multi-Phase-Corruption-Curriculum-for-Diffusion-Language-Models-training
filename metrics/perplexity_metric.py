import math

from metrics.loss_metric import LossMetric
from metrics.metirc_enum import METRICS


class PerplexityMetric(LossMetric):
    """exp(token-weighted NLL) on all corrupted input positions, before denoising."""

    result_key = "perplexity"

    def __init__(self, config=None):
        super().__init__(config)
        self.name = METRICS.PERPLEXITY

    def compute(self, predictions=None, references=None):
        mean_nll = super().compute(predictions, references)
        if mean_nll is None:
            return None
        try:
            return math.exp(mean_nll)
        except OverflowError:
            return float("inf")
