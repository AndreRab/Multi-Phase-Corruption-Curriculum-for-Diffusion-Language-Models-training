from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class CorruptionOutput:
    corrupted_ids: torch.Tensor
    corrupted_positions: torch.Tensor

class CorruptionMethod(ABC):
    @abstractmethod
    def __call__(
        self,
        clean_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> CorruptionOutput:
        pass
    
    def set_epoch(self, epoch: int) -> None:
        pass
    