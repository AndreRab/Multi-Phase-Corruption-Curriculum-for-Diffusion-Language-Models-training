from corruption_utils import CorruptionMethod, CorruptionOutput
import torch

class MaskTokenCorruption(CorruptionMethod):
    def __init__(
        self,
        mask_token_id: int,
        num_diffusion_steps: int,
        minimum_probability: float = 0.01,
        maximum_probability: float = 0.95,
    ) -> None:
        self.mask_token_id = mask_token_id
        self.num_diffusion_steps = num_diffusion_steps
        self.minimum_probability = minimum_probability
        self.maximum_probability = maximum_probability

    def __str__(self) -> str:
        return 'mask'

    def __call__(
        self,
        clean_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> CorruptionOutput:
        corrupted_ids = clean_ids.clone()
        normalized_t = timesteps.float() / (self.num_diffusion_steps - 1)
        
        probabilities = (self.minimum_probability + normalized_t * (self.maximum_probability - self.minimum_probability))
        probabilities = probabilities.unsqueeze(1)

        random_values = torch.rand(
            clean_ids.shape,
            device=clean_ids.device,
        )
        corrupted_positions = random_values < probabilities

        corrupted_positions &= attention_mask.bool()

        if not corrupted_positions.any():
            valid_positions = attention_mask.bool().nonzero(as_tuple=False)
            if valid_positions.numel() == 0:
                return CorruptionOutput(
                    corrupted_ids=clean_ids,
                    corrupted_positions=torch.zeros_like(clean_ids, dtype=torch.bool),
                )
            random_index = torch.randint(
                low=0,
                high=valid_positions.size(0),
                size=(1,),
                device=clean_ids.device,
            )
            batch_idx, token_idx = valid_positions[random_index].squeeze(0)
            corrupted_positions[batch_idx, token_idx] = True

        corrupted_ids[corrupted_positions] = self.mask_token_id

        return CorruptionOutput(
            corrupted_ids=corrupted_ids,
            corrupted_positions=corrupted_positions,
        )