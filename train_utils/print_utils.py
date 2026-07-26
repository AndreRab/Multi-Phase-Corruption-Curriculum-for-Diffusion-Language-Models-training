import torch
from transformers import AutoTokenizer

def print_reconstructed(
    tokenizer: AutoTokenizer,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    logits: torch.Tensor,
    corrupted_positions: torch.Tensor,
):
    predicted_ids = logits.argmax(dim=-1)
    reconstructed_ids = corrupted_ids.clone()
    reconstructed_ids[corrupted_positions] = predicted_ids[corrupted_positions]

    for i in range(clean_ids.size(0)):
        print("Original:     ", tokenizer.decode(clean_ids[i]).split('<|pad|>')[0])
        print("Corrupted:    ", tokenizer.decode(corrupted_ids[i]).split('<|pad|>')[0])
        print("Reconstructed:", tokenizer.decode(reconstructed_ids[i]).split('<|pad|>')[0])
        print('-----------')