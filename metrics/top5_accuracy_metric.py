from metrics.accuracy_metric import AccuracyMetric
from metrics.metirc_enum import METRICS


class Top5AccuracyMetric(AccuracyMetric):
    stage = "commitment"
    result_key = "top5_accuracy_denoising"

    def __init__(self, config=None):
        super().__init__(config)
        self.name = METRICS.TOP_5_ACCURACY

    def update(self, logits, references):
        candidates = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        self.total += int((candidates == references[:, None]).any(dim=-1).sum().item())
        self.count += references.numel()
