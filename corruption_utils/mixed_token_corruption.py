from corruption_utils import CorruptionMethod, CorruptionOutput
import torch

class MixedTokenCorruption(CorruptionMethod):
    def __init__(
        self,
        corruption_methods: list[CorruptionMethod],
        iterations_intervals: list[int],
    ) -> None:
        if len(corruption_methods) != len(iterations_intervals):
            raise ValueError("corruption_methods and iterations_intervals must have the same length.")

        self.corruption_methods = corruption_methods
        self.iterations_intervals = iterations_intervals
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def current_method(self) -> CorruptionMethod:
        total_iterations_sum = 0
        for i, interval in enumerate(self.iterations_intervals):
            total_iterations_sum += interval
            if self.current_epoch < total_iterations_sum:
                return self.corruption_methods[i]

        return self.corruption_methods[-1]

    def method_names(self) -> list[str]:
        """Return method names in the same order as the training intervals."""
        return [str(method) for method in self.corruption_methods]

    def __str__(self) -> str:
        return str(self.current_method())

    def __call__(
        self,
        clean_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> CorruptionOutput:
        corruption_method = self.current_method()

        return corruption_method(
            clean_ids=clean_ids,
            attention_mask=attention_mask,
            timesteps=timesteps,
        )
