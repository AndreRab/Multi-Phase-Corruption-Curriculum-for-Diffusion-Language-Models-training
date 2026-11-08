from enum import Enum

class METRICS(Enum):
    LOSS = "loss"
    MASKED_PERPLEXITY = "masked_perplexity"
    ACCURACY = "accuracy"
    DENOISING_ACCURACY = "denoising_accuracy"
    BLEU_4 = "bleu_4"
    ROUGE_L = "rouge_l"
    TOP_5_ACCURACY = "top_5_accuracy"
    MAUVE = "mauve"
