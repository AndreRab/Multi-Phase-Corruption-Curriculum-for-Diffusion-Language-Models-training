import torch
from transformers import AutoTokenizer

from corruption_utils import CorruptionMethod
        
class DiffusionDataCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        corruption_method: CorruptionMethod,
        num_diffusion_steps: int,
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.corruption_method = corruption_method
        self.num_diffusion_steps = num_diffusion_steps
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        texts = [example["text"] for example in examples]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        clean_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        valid_sample_indices = (attention_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

        if len(valid_sample_indices) == 0:
            return {
                "clean_ids": torch.empty(0, 0, dtype=clean_ids.dtype),
                "corrupted_ids": torch.empty(0, 0, dtype=clean_ids.dtype),
                "corrupted_positions": torch.empty(0, 0, dtype=torch.bool),
                "attention_mask": torch.empty(0, 0, dtype=attention_mask.dtype),
                "timesteps": torch.empty(0, dtype=torch.long),
            }

        clean_ids = clean_ids[valid_sample_indices]
        attention_mask = attention_mask[valid_sample_indices]

        batch_size = clean_ids.size(0)

        timesteps = torch.randint(
            low=0,
            high=self.num_diffusion_steps,
            size=(batch_size,),
        )

        corruption = self.corruption_method(
            clean_ids=clean_ids,
            attention_mask=attention_mask,
            timesteps=timesteps,
        )

        return {
            "clean_ids": clean_ids,
            "corrupted_ids": corruption.corrupted_ids,
            "corrupted_positions": corruption.corrupted_positions,
            "attention_mask": attention_mask,
            "timesteps": timesteps,
        }