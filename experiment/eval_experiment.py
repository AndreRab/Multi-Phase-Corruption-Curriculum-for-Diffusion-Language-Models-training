import json
import os
from pathlib import Path
from tqdm import tqdm

from experiment.eval_experiment_config import EvalExperimentConfig


def _setup_distributed(torch):
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    return distributed, rank, device


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


def _evaluate_models(models, batches, device):
    import torch

    total_loss = [0.0 for _ in models]
    total_correct = [0.0 for _ in models]
    total_positions = [0 for _ in models]

    for model in models:
        model.eval()

    with torch.no_grad():
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            if batch["clean_ids"].numel() == 0:
                continue

            for model_idx, model in enumerate(models):
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
                total_loss[model_idx] += loss.item() * position_count
                total_correct[model_idx] += correct
                total_positions[model_idx] += position_count

    return total_loss, total_correct, total_positions

def _evaluate_models_denoising(models, batches, device, num_iterations):
    import torch

    total_correct = [0.0 for _ in models]
    total_positions = [0 for _ in models]

    for model in models:
        model.eval()

    with torch.no_grad():
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            if batch["clean_ids"].numel() == 0:
                continue

            for model_idx, model in enumerate(models):
                reconstructed_ids = model.denoise(
                    corrupted_ids=batch["corrupted_ids"],
                    attention_mask=batch["attention_mask"],
                    corrupted_positions=batch["corrupted_positions"],
                    num_iterations=num_iterations,
                )
                positions = (
                    batch["corrupted_positions"].bool()
                    & batch["attention_mask"].bool()
                )
                position_count = int(positions.sum().item())
                if position_count == 0:
                    continue

                correct = int(
                    (reconstructed_ids[positions] == batch["clean_ids"][positions])
                    .sum()
                    .item()
                )
                total_correct[model_idx] += correct
                total_positions[model_idx] += position_count

    return total_correct, total_positions

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
    models = [
        GPT2DiffusionTransformer.from_file_path(
            file_path=model_path,
            model_name=config.model_name,
            num_diffusion_steps=config.num_diffusion_steps,
            vocabulary_size=len(tokenizer),
            device=device,
        ).to(device)
        for model_path in config.models_path
    ]

    for method_name, rate in tqdm(config.corruption_grid, desc=f"Evaluating corruption methods"):
        corruption = _build_corruption(method_name, rate, models[0], tokenizer, config)
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

        # Reuse the same corrupted batches for one-step and iterative evaluation.
        batches = list(loader)
        loss_sum, correct, positions = _evaluate_models(models, batches, device)
        correct_denoising, denoising_positions = _evaluate_models_denoising(
            models,
            batches,
            device,
            config.denoise_iterations,
        )
        if distributed:
            one_step_values = torch.tensor(
                [loss_sum, correct, positions],
                dtype=torch.float64,
                device=device,
            )
            denoising_values = torch.tensor(
                [correct_denoising, denoising_positions],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(one_step_values, op=dist.ReduceOp.SUM)
            dist.all_reduce(denoising_values, op=dist.ReduceOp.SUM)
            loss_sum, correct, positions = one_step_values.tolist()
            correct_denoising, denoising_positions = denoising_values.tolist()

        for model_path, loss_one, correct_one, positions_one, correct_denoising_one, positions_denoising_one in zip(
            config.models_path,
            loss_sum,
            correct,
            positions,
            correct_denoising,
            denoising_positions,
        ):
            results.append({
                "model": Path(model_path).stem,
                "model_path": model_path,
                "dataset": config.dataset_name if config.dataset_path is None else f"{config.dataset_path}/{config.dataset_name}",
                "split": config.dataset_split,
                "corruption_method": method_name,
                "corruption_rate": rate,
                "loss": loss_one / positions_one if positions_one else None,
                "accuracy": correct_one / positions_one if positions_one else None,
                "num_positions": int(positions_one),
                "accuracy_denoising": (
                    correct_denoising_one / positions_denoising_one
                    if positions_denoising_one
                    else None
                ),
                "denoise_iterations": config.denoise_iterations,
                "num_denoising_positions": int(positions_denoising_one),
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
