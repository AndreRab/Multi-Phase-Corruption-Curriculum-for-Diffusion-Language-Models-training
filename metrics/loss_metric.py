from metrics.accuracy_metric import AccuracyMetric
from metrics.metirc_enum import METRICS


class LossMetric(AccuracyMetric):
    result_key = "loss"

    def __init__(self, config=None):
        super().__init__(config)
        self.name = METRICS.LOSS

    def update(self, logits, references):
        import torch.nn.functional as F
        if references.numel():
            self.total += float(F.cross_entropy(logits, references, reduction="sum").item())
            self.count += references.numel()
