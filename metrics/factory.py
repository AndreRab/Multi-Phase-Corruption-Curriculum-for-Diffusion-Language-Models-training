from metrics.metirc_enum import METRICS
from metrics.accuracy_metric import AccuracyMetric
from metrics.denoising_accuracy_metric import DenoisingAccuracyMetric
from metrics.loss_metric import LossMetric
from metrics.top5_accuracy_metric import Top5AccuracyMetric
from metrics.bleu4_metric import Bleu4Metric
from metrics.rouge_l_metric import RougeLMetric
from metrics.mauve_metric import MauveMetric

METRIC_CLASSES = {
    METRICS.LOSS: LossMetric,
    METRICS.ACCURACY: AccuracyMetric,
    METRICS.DENOISING_ACCURACY: DenoisingAccuracyMetric,
    METRICS.TOP_5_ACCURACY: Top5AccuracyMetric,
    METRICS.BLEU_4: Bleu4Metric,
    METRICS.ROUGE_L: RougeLMetric,
    METRICS.MAUVE: MauveMetric,
}


def parse_metric(value):
    if isinstance(value, METRICS):
        return value
    aliases = {cls.result_key: name for name, cls in METRIC_CLASSES.items()}
    if isinstance(value, str) and value in aliases:
        return aliases[value]
    return METRICS(value)


def build_metrics(names, config=None):
    return [METRIC_CLASSES[name](config)
            for name in dict.fromkeys(parse_metric(value) for value in names)]


def metric_keys(names):
    return {METRIC_CLASSES[parse_metric(value)].result_key for value in names}
