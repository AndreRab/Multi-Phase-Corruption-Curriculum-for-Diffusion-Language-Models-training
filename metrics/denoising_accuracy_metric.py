from metrics.accuracy_metric import AccuracyMetric
from metrics.metirc_enum import METRICS


class DenoisingAccuracyMetric(AccuracyMetric):
    stage = "denoising"
    result_key = "accuracy_denoising"

    def __init__(self, config=None):
        super().__init__(config)
        self.name = METRICS.DENOISING_ACCURACY
