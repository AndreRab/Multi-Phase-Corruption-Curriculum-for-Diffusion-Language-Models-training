"""Shared decoding and orchestration; scoring lives in individual metric classes."""
from metrics.factory import build_metrics


class ReconstructionMetrics:
    def __init__(self, config):
        self.metrics = [metric for metric in build_metrics(config.metrics, config)
                        if metric.requires_text]

    def compute(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must be paired.")
        if not self.metrics:
            return {}
        result = {"num_texts": len(predictions)}
        for metric in self.metrics:
            result[metric.result_key] = metric.compute(predictions, references)
            result.update(getattr(metric, "metadata", {}))
        return result


def decode_valid_text(tokenizer, ids, attention_mask):
    """Remove padding by attention mask, but preserve generated special tokens."""
    return [tokenizer.decode(row[mask.bool()].tolist(), skip_special_tokens=False,
                             clean_up_tokenization_spaces=False)
            for row, mask in zip(ids, attention_mask)]
