from corruption_utils import CorruptionMethod, CorruptionOutput
import torch

class MixedTokenCorruption(CorruptionMethod):
    def __init__(
        self,
        corruption_methods: list[CorruptionMethod],
        iterations_intervals: list[int],
    ) -> None:
        self.corruption_methods = corruption_methods
        self.iterations_intervals = iterations_intervals
        self.counter = 0

    def __str__(self) -> str:
        total_iterations_sum = 0
        for i in range(len(self.iterations_intervals)):
            total_iterations_sum += self.iterations_intervals[i]
            if self.counter < total_iterations_sum:
                return str(self.corruption_methods[i])
        return str(self.corruption_methods[-1])

    def __call__(
        self,
        clean_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> CorruptionOutput:
        corruption_method = self.corruption_methods[-1]

        for i in range(len(self.iterations_intervals)):
            if self.counter < sum(self.iterations_intervals[:i + 1]):
                corruption_method = self.corruption_methods[i]
                break

        self.counter += 1

        return corruption_method(
            clean_ids=clean_ids,
            attention_mask=attention_mask,
            timesteps=timesteps,
        )