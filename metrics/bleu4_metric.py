from metrics.base_metric import BaseMetric
from metrics.metirc_enum import METRICS


class Bleu4Metric(BaseMetric):
    requires_text = True
    result_key = "bleu4"

    def __init__(self, config=None):
        super().__init__(METRICS.BLEU_4)
        self.metadata = {}

    def compute(self, predictions, references):
        if not predictions:
            return None
        from sacrebleu.metrics import BLEU
        bleu = BLEU(max_ngram_order=4, tokenize="13a", smooth_method="exp", effective_order=False)
        value = bleu.corpus_score(predictions, [references]).score / 100
        self.metadata = {"bleu_signature": str(bleu.get_signature())}
        return value
