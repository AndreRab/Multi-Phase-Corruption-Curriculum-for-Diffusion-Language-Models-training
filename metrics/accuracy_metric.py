from metrics.base_metric import BaseMetric
from metrics.metirc_enum import METRICS


class AccuracyMetric(BaseMetric):
    stage = "one_step"
    result_key = "accuracy"

    def __init__(self, config=None):
        super().__init__(METRICS.ACCURACY)
        self.total = 0
        self.count = 0

    def update(self, predictions, references):
        self.total += int((predictions == references).sum().item())
        self.count += references.numel()

    def compute(self, predictions=None, references=None):
        total = self.total if predictions is None else predictions
        count = self.count if references is None else references
        return total / count if count else None
