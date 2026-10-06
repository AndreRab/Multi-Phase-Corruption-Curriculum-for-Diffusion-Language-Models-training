import json
import os
from pathlib import Path
from tqdm import tqdm

from experiments.eval_experiment_config import EvalExperimentConfig
from metrics.factory import build_metrics, metric_keys


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
    from corruption_utils import build_corruption

    return build_corruption(
        name,
        model=model,
        tokenizer=tokenizer,
        num_diffusion_steps=config.num_diffusion_steps,
        minimum_probability=config.corruption_minimum_probability,
        maximum_probability=rate,
        similar_number_of_neighbors=config.similar_number_of_neighbors,
    )


def _evaluate_models(models, batches, device, metrics=("loss", "accuracy")):
    import torch

    metric_instances = [{metric.result_key: metric for metric in build_metrics(metrics)} for _ in models]
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

                if "loss" in metric_instances[model_idx]:
                    metric = metric_instances[model_idx]["loss"]
                    metric.update(logits[positions], batch["clean_ids"][positions])
                    total_loss[model_idx] = metric.total
                if "accuracy" in metric_instances[model_idx]:
                    metric = metric_instances[model_idx]["accuracy"]
                    metric.update(logits[positions].argmax(dim=-1), batch["clean_ids"][positions])
                    total_correct[model_idx] = metric.total
                total_positions[model_idx] += position_count

    return total_loss, total_correct, total_positions

def _evaluate_models_denoising(models, batches, device, num_iterations, tokenizer, metrics=None):
    import torch

    total_correct = [0.0 for _ in models]
    total_positions = [0 for _ in models]
    top5_hits = [0 for _ in models]
    top5_positions = [0 for _ in models]
    predictions = [[] for _ in models]
    references = [[] for _ in models]
    from metrics.text_metrics import decode_valid_text
    selected = metric_keys(metrics) if metrics is not None else {"accuracy_denoising", "top5_accuracy_denoising", "bleu4", "rouge_l_f1"}
    metric_instances = [{metric.result_key: metric for metric in build_metrics(selected)} for _ in models]
    collect_text = bool(selected & {"bleu4", "rouge_l_f1", "mauve"})

    for model in models:
        model.eval()

    with torch.no_grad():
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            if batch["clean_ids"].numel() == 0:
                continue

            for model_idx, model in enumerate(models):
                def on_commit(batch_idx, selected_positions, logits):
                    metric = metric_instances[model_idx]["top5_accuracy_denoising"]
                    metric.update(logits, batch["clean_ids"][batch_idx, selected_positions])
                    top5_hits[model_idx] = metric.total
                    top5_positions[model_idx] = metric.count

                reconstructed_ids = model.denoise(
                    corrupted_ids=batch["corrupted_ids"],
                    attention_mask=batch["attention_mask"],
                    corrupted_positions=batch["corrupted_positions"],
                    num_iterations=num_iterations,
                    on_commit=on_commit if "top5_accuracy_denoising" in selected else None,
                )
                if collect_text:
                    predictions[model_idx].extend(decode_valid_text(tokenizer, reconstructed_ids, batch["attention_mask"]))
                    references[model_idx].extend(decode_valid_text(tokenizer, batch["clean_ids"], batch["attention_mask"]))
                positions = (
                    batch["corrupted_positions"].bool()
                    & batch["attention_mask"].bool()
                )
                position_count = int(positions.sum().item())
                if position_count == 0:
                    continue

                if "accuracy_denoising" in selected:
                    metric = metric_instances[model_idx]["accuracy_denoising"]
                    metric.update(reconstructed_ids[positions], batch["clean_ids"][positions])
                    total_correct[model_idx] = metric.total
                total_positions[model_idx] += position_count

    return total_correct, total_positions, top5_hits, top5_positions, predictions, references

def run_eval_experiment(config: EvalExperimentConfig) -> None:
    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from model_wrappers import GPT2DiffusionTransformer
    from train_utils import DiffusionDataCollator
    from experiments.reproducibility import seed_everything

    distributed, rank, device = _setup_distributed(torch)
    seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>", "mask_token": "<|mask|>"})

    dataset = load_dataset(config.dataset_name) if config.dataset_path is None else load_dataset(
        config.dataset_path,
        config.dataset_name,
    )
    dataset = _select_non_empty_text_rows(dataset[config.dataset_split])
    # Disjoint shards avoid DistributedSampler padding/duplicating test examples.
    sampler = list(range(rank, len(dataset), dist.get_world_size())) if distributed else None
    from metrics.text_metrics import ReconstructionMetrics
    metrics = ReconstructionMetrics(config)

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

    # Model construction may consume random numbers when a checkpoint is
    # initialized before loading. Reset the evaluation stream afterwards so
    # the same seed produces the same corrupted benchmark across K configs.
    seed_everything(config.seed)

    for method_name, rate in tqdm(config.corruption_grid, desc=f"Evaluating corruption methods"):
        corruption = _build_corruption(method_name, rate, models[0], tokenizer, config)
        collator = DiffusionDataCollator(
            tokenizer=tokenizer,
            corruption_method=corruption,
            num_diffusion_steps=config.num_diffusion_steps,
            max_length=config.max_length,
            fixed_timestep=config.num_diffusion_steps - 1,
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
        loss_sum, correct, positions = ([0] * len(models) for _ in range(3))
        if metric_keys(config.metrics) & {"loss", "accuracy"}:
            loss_sum, correct, positions = _evaluate_models(models, batches, device, config.metrics)
        correct_denoising, denoising_positions, top5_hits, top5_positions = ([0] * len(models) for _ in range(4))
        predictions, references = ([[] for _ in models] for _ in range(2))
        if metric_keys(config.metrics) - {"loss", "accuracy"}:
            correct_denoising, denoising_positions, top5_hits, top5_positions, predictions, references = _evaluate_models_denoising(
                models, batches, device, config.denoise_iterations, tokenizer, config.metrics,
            )
        if distributed:
            one_step_values = torch.tensor(
                [loss_sum, correct, positions],
                dtype=torch.float64,
                device=device,
            )
            denoising_values = torch.tensor(
                [correct_denoising, denoising_positions, top5_hits, top5_positions],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(one_step_values, op=dist.ReduceOp.SUM)
            dist.all_reduce(denoising_values, op=dist.ReduceOp.SUM)
            loss_sum, correct, positions = one_step_values.tolist()
            correct_denoising, denoising_positions, top5_hits, top5_positions = denoising_values.tolist()
            gathered = [None] * dist.get_world_size() if rank == 0 else None
            dist.gather_object((predictions, references), gathered, dst=0)
            if rank == 0:
                predictions = [sum((item[0][idx] for item in gathered), []) for idx in range(len(models))]
                references = [sum((item[1][idx] for item in gathered), []) for idx in range(len(models))]

        if rank != 0:
            continue
        text_results = [metrics.compute(pred, ref) for pred, ref in zip(predictions, references)]

        scalar_metrics = [metric for metric in build_metrics(config.metrics, config) if not metric.requires_text]
        for model_idx, (model_path, loss_one, correct_one, positions_one, correct_denoising_one, positions_denoising_one) in enumerate(zip(
            config.models_path,
            loss_sum,
            correct,
            positions,
            correct_denoising,
            denoising_positions,
        )):
            row = {
                "model": Path(model_path).stem,
                "model_path": model_path,
                "dataset": config.dataset_name if config.dataset_path is None else f"{config.dataset_path}/{config.dataset_name}",
                "split": config.dataset_split,
                "corruption_method": method_name,
                "corruption_rate": rate,
                "seed": config.seed,
                "initial_timestep": config.num_diffusion_steps - 1,
                "loss": loss_one / positions_one if positions_one else None,
                "accuracy": correct_one / positions_one if positions_one else None,
                "num_positions": int(positions_one),
                "accuracy_denoising": (
                    correct_denoising_one / positions_denoising_one
                    if positions_denoising_one
                    else None
                ),
                "top5_accuracy_denoising": top5_hits[model_idx] / top5_positions[model_idx] if top5_positions[model_idx] else None,
                "num_top5_positions": int(top5_positions[model_idx]),
                **text_results[model_idx],
                "denoise_iterations": config.denoise_iterations,
                "num_denoising_positions": int(positions_denoising_one),
            }
            statistics = {
                "loss": (loss_one, positions_one),
                "accuracy": (correct_one, positions_one),
                "accuracy_denoising": (correct_denoising_one, positions_denoising_one),
                "top5_accuracy_denoising": (top5_hits[model_idx], top5_positions[model_idx]),
            }
            for metric in scalar_metrics:
                row[metric.result_key] = metric.compute(*statistics[metric.result_key])
            for name in {"loss", "accuracy", "accuracy_denoising", "top5_accuracy_denoising"} - metric_keys(config.metrics):
                row.pop(name, None)
            if "top5_accuracy_denoising" not in config.metrics:
                row.pop("num_top5_positions", None)
            if not metric_keys(config.metrics) & {"loss", "accuracy"}:
                row.pop("num_positions", None)
            if not metric_keys(config.metrics) - {"loss", "accuracy"}:
                row.pop("num_denoising_positions", None)
                row.pop("denoise_iterations", None)
            row["metrics"] = [name.value for name in config.metrics]
            results.append(row)

    if rank == 0:
        output_path = Path(config.result_folder) / config.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as output_file:
            json.dump(results, output_file, indent=4)
        print(f"Saved evaluation results to {output_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
