from corruption_utils import CorruptionMethod, CorruptionOutput
import torch
import torch.nn.functional as F

class SimilarTokenCorruption(CorruptionMethod):
    def __init__(
        self,
        num_diffusion_steps: int,
        embedding_weight: torch.Tensor,
        number_of_neighbors: int = 20,
        minimum_probability: float = 0.01,
        maximum_probability: float = 0.95,
        chunk_size: int = 1024,
    ) -> None:
        self.num_diffusion_steps = num_diffusion_steps
        self.minimum_probability = minimum_probability
        self.maximum_probability = maximum_probability
        self.number_of_neighbors = number_of_neighbors

        self.nearest_token_ids = self._build_nearest_token_table(
            embedding_weight=embedding_weight,
            number_of_neighbors=number_of_neighbors,
            chunk_size=chunk_size,
        )

    def __str__(self, ) -> str:
        return 'similar'

    @staticmethod
    @torch.no_grad()
    def _build_nearest_token_table(
        embedding_weight: torch.Tensor,
        number_of_neighbors: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Buduje tabelę najbliższych tokenów według cosine similarity.

        Zwraca:
            Tensor o kształcie:
            [vocabulary_size, number_of_neighbors]
        """
        if embedding_weight.ndim != 2:
            raise ValueError(
                "embedding_weight must have shape "
                "[vocabulary_size, embedding_size]"
            )

        if number_of_neighbors <= 0:
            raise ValueError(
                "number_of_neighbors must be greater than zero"
            )

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        vocabulary_size = embedding_weight.size(0)

        if number_of_neighbors >= vocabulary_size:
            raise ValueError(
                "number_of_neighbors must be smaller "
                "than vocabulary size"
            )

        device = embedding_weight.device

        normalized_embeddings = F.normalize(
            embedding_weight.detach().float(),
            p=2,
            dim=-1,
        )

        nearest_chunks = []

        for start in (range(0, vocabulary_size, chunk_size)):
            end = min(
                start + chunk_size,
                vocabulary_size,
            )

            current_embeddings = normalized_embeddings[
                start:end
            ]

            similarities = (
                current_embeddings
                @ normalized_embeddings.T
            )

            local_row_indices = torch.arange(
                end - start,
                device=device,
            )

            global_token_indices = torch.arange(
                start,
                end,
                device=device,
            )

            # Token nie może być własnym sąsiadem.
            similarities[
                local_row_indices,
                global_token_indices,
            ] = -torch.inf

            nearest_ids = torch.topk(
                similarities,
                k=number_of_neighbors,
                dim=-1,
            ).indices

            nearest_chunks.append(
                nearest_ids.cpu()
            )

        return torch.cat(
            nearest_chunks,
            dim=0,
        )

    def __call__(
        self,
        clean_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> CorruptionOutput:
        corrupted_ids = clean_ids.clone()

        normalized_t = timesteps.float() / (
            self.num_diffusion_steps - 1
        )

        probabilities = (
            self.minimum_probability
            + normalized_t
            * (
                self.maximum_probability
                - self.minimum_probability
            )
        )

        probabilities = probabilities.unsqueeze(1)

        random_values = torch.rand(
            clean_ids.shape,
            device=clean_ids.device,
        )

        corrupted_positions = (
            random_values < probabilities
        )

        # Nie psujemy paddingu.
        corrupted_positions &= attention_mask.bool()

        # Gwarantujemy przynajmniej jedną korupcję w batchu.
        if not corrupted_positions.any():
            valid_positions = attention_mask.bool().nonzero(
                as_tuple=False
            )

            if valid_positions.numel() == 0:
                # If no valid positions to corrupt, return the original clean_ids
                # and an empty corrupted_positions mask for this batch item.
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

            batch_idx, token_idx = valid_positions[
                random_index
            ].squeeze(0)

            corrupted_positions[
                batch_idx,
                token_idx,
            ] = True

        original_token_ids = clean_ids[
            corrupted_positions
        ]

        nearest_token_ids = self.nearest_token_ids.to(
            clean_ids.device
        )

        candidate_neighbors = nearest_token_ids[
            original_token_ids
        ]

        number_of_corrupted_tokens = (
            original_token_ids.numel()
        )

        random_neighbor_indices = torch.randint(
            low=0,
            high=self.number_of_neighbors,
            size=(number_of_corrupted_tokens,),
            device=clean_ids.device,
        )

        row_indices = torch.arange(
            number_of_corrupted_tokens,
            device=clean_ids.device,
        )

        replacement_token_ids = candidate_neighbors[
            row_indices,
            random_neighbor_indices,
        ]

        corrupted_ids[
            corrupted_positions
        ] = replacement_token_ids

        return CorruptionOutput(
            corrupted_ids=corrupted_ids,
            corrupted_positions=corrupted_positions,
        )