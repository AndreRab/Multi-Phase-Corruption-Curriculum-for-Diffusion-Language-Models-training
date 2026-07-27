import json
from dataclasses import asdict
from pathlib import Path

from experiment.base_experiment_config import ExperimentConfig


def run_experiment(config: ExperimentConfig) -> None:
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from corruption_utils import (
        MaskTokenCorruption,
        MixedTokenCorruption,
        RandomTokenCorruption,
        SimilarTokenCorruption,
    )
    from model_wrappers import GPT2DiffusionTransformer
    from train_utils import DiffusionDataCollator, TrainingOutput, train

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

    ds = load_dataset(config.dataset_path, config.dataset_name)

    dataset_test = ds["test"]
    dataset_train = ds["train"]

    corruption_method_1 = SimilarTokenCorruption(
        embedding_weight=model.transformer.wte.weight,
        num_diffusion_steps=config.num_diffusion_steps,
        number_of_neighbors=20,
        minimum_probability=0.01,
        maximum_probability=0.30,
    )

    corruption_method_2 = RandomTokenCorruption(
        dictionary_size=len(tokenizer),
        num_diffusion_steps=config.num_diffusion_steps,
        minimum_probability=0.01,
        maximum_probability=0.95,
    )

    corruption_method_3 = MaskTokenCorruption(
        mask_token_id=tokenizer.mask_token_id,
        num_diffusion_steps=config.num_diffusion_steps,
        minimum_probability=0.01,
        maximum_probability=0.95,
    )

    corruption_method = MixedTokenCorruption(
        corruption_methods=[
            corruption_method_1,
            corruption_method_2,
            corruption_method_3,
        ],
        iterations_intervals=config.iterations_intervals,
    )

    collator = DiffusionDataCollator(
        tokenizer=tokenizer,
        corruption_method=corruption_method,
        num_diffusion_steps=config.num_diffusion_steps,
        max_length=config.max_length,
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    training_output: TrainingOutput = train(
        device,
        model,
        optimizer,
        sum(config.iterations_intervals),
        train_loader,
        test_loader,
    )

    Path(config.model_save_path).mkdir(parents=True, exist_ok=True)
    Path(config.training_output_save_path).mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), f"{config.model_save_path}/{config.model_id}.pt")
    with open(f"{config.training_output_save_path}/{config.train_output_id}.json", "w") as f:
        json.dump(asdict(training_output), f, indent=4)
