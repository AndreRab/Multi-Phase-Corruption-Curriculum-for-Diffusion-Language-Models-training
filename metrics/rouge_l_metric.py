from metrics.base_metric import BaseMetric
from metrics.metirc_enum import METRICS

class RougeLMetric(BaseMetric):
    requires_text = True
    result_key = "rouge_l_f1"

    def __init__(self, config=None):
        super().__init__(METRICS.ROUGE_L)

    def compute(self, predictions, references):
        if not predictions:
            return None
        from rouge_score.rouge_scorer import RougeScorer
        scorer = RougeScorer(["rougeL"], use_stemmer=False)
        return sum(scorer.score(ref, pred)["rougeL"].fmeasure
                   for pred, ref in zip(predictions, references)) / len(predictions)
