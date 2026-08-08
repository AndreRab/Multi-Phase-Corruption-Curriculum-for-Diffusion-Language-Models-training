import json
import os
from pathlib import Path

from experiment.eval_experiment_config import EvalExperimentConfig


def _setup_distributed(torch):
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return distributed, int(os.environ.get("RANK", "0")), device


def _select_non_empty_text_rows(dataset):
    if "text" not in dataset.column_names:
        raise ValueError(
            f"Evaluation dataset must contain a 'text' column; found {dataset.column_names}."
        )
    indices = [idx for idx, example in enumerate(dataset) if example["text"].strip()]
    return dataset.select(indices)


def _build_corruption(name, rate, model, tokenizer, config):
    from corruption_utils import (
        MaskTokenCorruption,
        RandomTokenCorruption,
        SimilarTokenCorruption,
    )

    if name == "similar":
        return SimilarTokenCorruption(
            embedding_weight=model.transformer.wte.weight,
            num_diffusion_steps=config.num_diffusion_steps,
            number_of_neighbors=20,
            minimum_probability=0.01,
            maximum_probability=rate,
        )
    if name == "mask":
        return MaskTokenCorruption(
            mask_token_id=tokenizer.mask_token_id,
            num_diffusion_steps=config.num_diffusion_steps,
            minimum_probability=0.01,
            maximum_probability=rate,
        )
    if name == "random":
        return RandomTokenCorruption(
            dictionary_size=len(tokenizer),
            num_diffusion_steps=config.num_diffusion_steps,
            minimum_probability=0.01,
            maximum_probability=rate,
        )
        
    raise ValueError(f"Unsupported corruption method: {name}")


def _evaluate_model(model, loader, device):
    import torch

    total_loss = 0.0
    total_correct = 0.0
    total_positions = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if batch["clean_ids"].numel() == 0:
                continue

            logits = model(
                corrupted_ids=batch["corrupted_ids"],
                timesteps=batch["timesteps"],
                attention_mask=batch["attention_mask"],
            )
            positions = batch["corrupted_positions"].bool() & batch["attention_mask"].bool()
            position_count = int(positions.sum().item())
            if position_count == 0:
                continue

            loss = model.compute_loss(
                logits, batch["clean_ids"], batch["corrupted_positions"], batch["attention_mask"]
            )
            predictions = logits.argmax(dim=-1)
            correct = int((predictions[positions] == batch["clean_ids"][positions]).sum().item())
            total_loss += loss.item() * position_count
            total_correct += correct
            total_positions += position_count

    return total_loss, total_correct, total_positions


def run_eval_experiment(config: EvalExperimentConfig) -> None:
    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer

    from model_wrappers import GPT2DiffusionTransformer
    from train_utils import DiffusionDataCollator

    distributed, rank, device = _setup_distributed(torch)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>", "mask_token": "<|mask|>"})

    dataset = load_dataset(config.dataset_name) if config.dataset_path is None else load_dataset(
        config.dataset_path,
        config.dataset_name,
    )
    dataset = _select_non_empty_text_rows(dataset[config.dataset_split])
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None

    results = []
    for model_path in config.models_path:
        model = GPT2DiffusionTransformer.from_file_path(
            file_path=model_path,
            model_name=config.model_name,
            num_diffusion_steps=config.num_diffusion_steps,
            vocabulary_size=len(tokenizer),
            device=device,
        ).to(device)

        for method_name, rate in config.corruption_grid:
            corruption = _build_corruption(method_name, rate, model, tokenizer, config)
            collator = DiffusionDataCollator(
                tokenizer=tokenizer,
                corruption_method=corruption,
                num_diffusion_steps=config.num_diffusion_steps,
                max_length=config.max_length,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=sampler,
                collate_fn=collator,
            )
            loss_sum, correct, positions = _evaluate_model(model, loader, device)
            if distributed:
                values = torch.tensor([loss_sum, correct, positions], dtype=torch.float64, device=device)
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                loss_sum, correct, positions = values.tolist()

            results.append({
                "model": Path(model_path).stem,
                "model_path": model_path,
                "dataset": config.dataset_name if config.dataset_path is None else f"{config.dataset_path}/{config.dataset_name}",
                "split": config.dataset_split,
                "corruption_method": method_name,
                "corruption_rate": rate,
                "loss": loss_sum / positions if positions else None,
                "accuracy": correct / positions if positions else None,
                "num_positions": int(positions),
            })

    if rank == 0:
        output_path = Path(config.result_folder) / config.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as output_file:
            json.dump(results, output_file, indent=4)
        print(f"Saved evaluation results to {output_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
