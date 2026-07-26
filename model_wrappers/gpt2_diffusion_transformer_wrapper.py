import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from pathlib import Path

class GPT2DiffusionTransformer(nn.Module):
    def __init__(
        self,
        model_name: str = "openai-community/gpt2",
        num_diffusion_steps: int = 1000,
        vocabulary_size: int | None = None,
    ) -> None:
        super().__init__()

        pretrained = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
        )

        if vocabulary_size is not None:
            pretrained.resize_token_embeddings(
                vocabulary_size
            )

        self.transformer = pretrained.transformer
        self.lm_head = pretrained.lm_head
        self.config = pretrained.config

        self.time_embedding = nn.Embedding(
            num_diffusion_steps,
            self.config.n_embd,
        )

        self.num_diffusion_steps = num_diffusion_steps

    def forward(
        self,
        corrupted_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = corrupted_ids.shape

        if sequence_length == 0:
            return torch.empty(
                batch_size,
                0,
                self.config.vocab_size,
                device=corrupted_ids.device,
                dtype=self.lm_head.weight.dtype
            )

        token_embeddings = self.transformer.wte(corrupted_ids)
        time_embeddings = self.time_embedding(timesteps).unsqueeze(1)
        input_embeddings = token_embeddings + time_embeddings

        bidirectional_mask = torch.zeros(
            batch_size,
            1,
            sequence_length,
            sequence_length,
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )

        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].bool()

            bidirectional_mask = (
                bidirectional_mask.masked_fill(
                    ~padding_mask,
                    torch.finfo(
                        input_embeddings.dtype
                    ).min,
                )
            )

        outputs = self.transformer(
            inputs_embeds=input_embeddings,
            attention_mask=bidirectional_mask,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def denoise(
        self,
        corrupted_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        corrupted_positions: torch.Tensor | None = None,
        num_iterations: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        reconstructed_ids = corrupted_ids.clone().to(device)

        if attention_mask is None:
            attention_mask = torch.ones_like(
                reconstructed_ids,
                dtype=torch.long,
                device=device,
            )
        else:
            attention_mask = attention_mask.to(device)

        if corrupted_positions is None:
            positions_to_update = attention_mask.bool()
        else:
            positions_to_update = (
                corrupted_positions.to(device).bool()
                & attention_mask.bool()
            )

        if not positions_to_update.any():
            return reconstructed_ids

        was_training = self.training
        self.eval()

        active_positions = positions_to_update.clone()

        for iteration in range(num_iterations):
            if not active_positions.any():
                break

            timestep = round((self.num_diffusion_steps - 1) * (1.0 - iteration / max(num_iterations - 1, 1)))
            timesteps = torch.full(
                size=(reconstructed_ids.size(0),),
                fill_value=timestep,
                dtype=torch.long,
                device=device,
            )

            logits = self.forward(
                corrupted_ids=reconstructed_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )
            logits = logits / temperature

            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_ids = (probabilities.max(dim=-1))

            for batch_idx in range(reconstructed_ids.size(0)):
                current_positions = active_positions[batch_idx].nonzero(as_tuple=True)[0]
                if current_positions.numel() == 0:
                    continue

                remaining_iterations = (num_iterations - iteration)
                number_to_update = max(1, int(current_positions.numel() / remaining_iterations))

                current_confidence = confidence[batch_idx, current_positions]
                selected_local_indices = torch.topk(current_confidence, k=min(number_to_update, current_positions.numel())).indices
                selected_positions = current_positions[selected_local_indices]

                reconstructed_ids[batch_idx, selected_positions] = predicted_ids[batch_idx, selected_positions]
                active_positions[batch_idx, selected_positions] = False

        if was_training:
            self.train()

        return reconstructed_ids

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        clean_ids: torch.Tensor,
        corrupted_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        labels = clean_ids.clone()

        if corrupted_positions is not None:
            labels[~corrupted_positions.bool()] = -100

        if attention_mask is not None:
            labels[attention_mask == 0] = -100

        valid_positions = labels != -100

        if not valid_positions.any():
            return logits.sum() * 0.0

        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

    @staticmethod
    def compute_accuracy(
        logits: torch.Tensor,
        clean_ids: torch.Tensor,
        corrupted_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> float: 
        predictions = logits.argmax(dim=-1)

        valid_positions = torch.ones_like(
            clean_ids,
            dtype=torch.bool,
        )

        if corrupted_positions is not None:
            valid_positions &= corrupted_positions.bool()

        if attention_mask is not None:
            valid_positions &= attention_mask.bool()

        if not valid_positions.any():
            return 0.0

        correct = (predictions[valid_positions] == clean_ids[valid_positions]).float()

        return correct.mean().item()
    
    @classmethod
    def from_file_path(
        cls,
        file_path: Path,
        model_name: str = "openai-community/gpt2",
        num_diffusion_steps: int = 1000,
        vocabulary_size: int | None = None,
        device: torch.device | str = "cpu",
    ) -> "GPT2DiffusionTransformer" | None:
        model = cls(
            model_name=model_name,
            num_diffusion_steps=num_diffusion_steps,
            vocabulary_size=vocabulary_size,
        )
        
        file_path = Path(file_path)
        if file_path.exists():
            print(f"Loading model from {file_path}")
            state_dict = torch.load(file_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)

        return model