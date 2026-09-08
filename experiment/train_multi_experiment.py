import json
import os
from dataclasses import asdict
from pathlib import Path

from experiment.train_multi_experiment_config import TrainMultiExperimentConfig


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

    rank = int(os.environ.get("RANK", "0"))
    return distributed, rank, local_rank, device


def _select_non_empty_text_rows(dataset):
    non_empty_indices = [
        idx
        for idx, example in enumerate(dataset)
        if example["text"].strip()
    ]
    return dataset.select(non_empty_indices)


def run_experiment(config: TrainMultiExperimentConfig) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer

    from corruption_utils import (
        MixedTokenCorruption,
        build_corruptions,
    )
    from model_wrappers import GPT2DiffusionTransformer
    from train_utils import DiffusionDataCollator, TrainingOutput, train

    distributed, rank, local_rank, device = _setup_distributed(torch)
    is_main_process = rank == 0

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.add_special_tokens({
        "pad_token": "<|pad|>",
        "mask_token": "<|mask|>",
    })

    model = GPT2DiffusionTransformer.from_file_path(
        file_path=f"{config.model_save_path}/{config.model_id}.pt",
        model_name=config.model_name,
        num_diffusion_steps=config.num_diffusion_steps,
        vocabulary_size=len(tokenizer),
    )
    model = model.to(device)

    ds = load_dataset(config.dataset_path, config.dataset_name)

    dataset_test = _select_non_empty_text_rows(ds["test"])
    dataset_train = _select_non_empty_text_rows(ds["train"])

    corruption_methods = build_corruptions(
        config.corruption_methods,
        model=model,
        tokenizer=tokenizer,
        num_diffusion_steps=config.num_diffusion_steps,
        minimum_probability=config.corruption_minimum_probability,
        maximum_probability=config.corruption_maximum_probability,
        similar_number_of_neighbors=config.similar_number_of_neighbors,
    )

    corruption_method = MixedTokenCorruption(
        corruption_methods=corruption_methods,
        iterations_intervals=config.iterations_intervals,
    )

    collator = DiffusionDataCollator(
        tokenizer=tokenizer,
        corruption_method=corruption_method,
        num_diffusion_steps=config.num_diffusion_steps,
        max_length=config.max_length,
    )

    train_sampler = (
        DistributedSampler(dataset_train, shuffle=True)
        if distributed
        else None
    )
    test_sampler = (
        DistributedSampler(dataset_test, shuffle=False)
        if distributed
        else None
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collator,
    )

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    training_output: TrainingOutput = train(
        device,
        model,
        optimizer,
        sum(config.iterations_intervals),
        train_loader,
        test_loader,
        is_main_process=is_main_process,
        train_diffusion_steps=config.train_diffusion_steps,
        rollout_loss_decay=config.rollout_loss_decay,
    )

    if is_main_process:
        Path(config.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(config.training_output_save_path).mkdir(parents=True, exist_ok=True)

        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), f"{config.model_save_path}/{config.model_id}.pt")
        with open(f"{config.training_output_save_path}/{config.train_output_id}.json", "w") as f:
            json.dump(asdict(training_output), f, indent=4)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
