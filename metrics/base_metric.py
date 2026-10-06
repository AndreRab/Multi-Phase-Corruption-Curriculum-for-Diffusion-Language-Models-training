from abc import ABC, abstractmethod


class BaseMetric(ABC):
    requires_text = False
    stage = "denoising"

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def compute(self, predictions, references):
        """Return the score from paired data or accumulated sufficient statistics."""
