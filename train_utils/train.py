from dataclasses import dataclass, field
import torch
import torch.distributed as dist
from tqdm import tqdm

from model_wrappers.gpt2_diffusion_transformer_wrapper import GPT2DiffusionTransformer
from corruption_utils.base_corruption_method import CorruptionMethod

@dataclass
class TrainingOutput:
    train_loss: list[float]
    test_loss: list[float]
    train_acc: list[float]
    test_acc: list[float]
    iterations_intervals: dict
    train_step_loss: list[list[float]] = field(default_factory=list)
    test_step_loss: list[list[float]] = field(default_factory=list)
    train_step_acc: list[list[float]] = field(default_factory=list)
    test_step_acc: list[list[float]] = field(default_factory=list)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _distributed_mean(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value

    value_tensor = torch.tensor(value, device=device)
    dist.all_reduce(value_tensor, op=dist.ReduceOp.AVG)
    return value_tensor.item()


def _prepare_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor] | None:
    batch = {key: value.to(device) for key, value in batch.items()}
    attention_mask = batch["attention_mask"]
    valid_samples = (attention_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    if len(valid_samples) == 0:
        return None
    return {key: value[valid_samples] for key, value in batch.items()}


def _rollout(
    model: GPT2DiffusionTransformer,
    batch: dict[str, torch.Tensor],
    num_steps: int,
    loss_decay: float,
) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
    """Run teacher-forced step 1 and then feed detached model predictions back in."""
    model_for_metrics = _unwrap_model(model)
    current_ids = batch["corrupted_ids"]
    step_losses: list[torch.Tensor] = []
    step_accuracies: list[float] = []

    for step in range(num_steps):
        logits = model(
            corrupted_ids=current_ids,
            timesteps=batch["timesteps"],
            attention_mask=batch["attention_mask"],
        )
        loss = model_for_metrics.compute_loss(
            logits=logits,
            clean_ids=batch["clean_ids"],
            corrupted_positions=batch["corrupted_positions"],
            attention_mask=batch["attention_mask"],
        )
        accuracy = model_for_metrics.compute_accuracy(
            logits=logits,
            clean_ids=batch["clean_ids"],
            corrupted_positions=batch["corrupted_positions"],
            attention_mask=batch["attention_mask"],
        )
        step_losses.append(loss)
        step_accuracies.append(accuracy)

        if step < num_steps - 1:
            with torch.no_grad():
                predicted_ids = logits.argmax(dim=-1)
                current_ids = current_ids.clone()
                positions = batch["corrupted_positions"].bool()
                current_ids[positions] = predicted_ids[positions]

    weights = [loss_decay ** step for step in range(num_steps)]
    total_loss = sum(weight * loss for weight, loss in zip(weights, step_losses))
    total_loss = total_loss / sum(weights)
    return total_loss, step_losses, step_accuracies


def train(
    device: torch.device, 
    model: GPT2DiffusionTransformer, 
    optimizer: torch.optim.Optimizer, 
    num_epochs: int, 
    train_loader: torch.utils.data.DataLoader, 
    test_loader: torch.utils.data.DataLoader = None,
    is_main_process: bool = True,
    train_diffusion_steps: int = 1,
    rollout_loss_decay: float = 0.5,
) -> TrainingOutput:
    model = model.to(device)
    model.train()

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    train_step_loss = []
    test_step_loss = []
    train_step_acc = []
    test_step_acc = []

    for epoch in range(num_epochs):
        train_sampler = getattr(train_loader, "sampler", None)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        corruption_method: CorruptionMethod = getattr(train_loader.collate_fn, "corruption_method", None)
        corruption_method.set_epoch(epoch)

        model.train()
        total_loss_train = 0.0
        total_loss_test = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
            dynamic_ncols=True,
            disable=not is_main_process,
        )

        for batch in progress_bar:
            batch = _prepare_batch(batch, device)
            if batch is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss, step_losses, step_accuracies = _rollout(
                model,
                batch,
                train_diffusion_steps,
                rollout_loss_decay,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            reduced_loss = _distributed_mean(loss.item(), device)
            reduced_accuracy = _distributed_mean(step_accuracies[-1], device)

            train_loss.append(reduced_loss)
            train_acc.append(reduced_accuracy)
            train_step_loss.append([_distributed_mean(item.item(), device) for item in step_losses])
            train_step_acc.append([_distributed_mean(item, device) for item in step_accuracies])

            total_loss_train += reduced_loss

            if is_main_process:
                progress_bar.set_postfix(
                    loss=f"{reduced_loss:.4f}",
                    acc=f"{reduced_accuracy:.4f}",
                    corruption_method=f"{str(train_loader.collate_fn.corruption_method)}"
                )

        if test_loader:
            was_training = model.training
            model.eval()

            for batch in tqdm(
                test_loader,
                desc=f"Testing {epoch + 1}/{num_epochs}",
                leave=False,
                dynamic_ncols=True,
                disable=not is_main_process,
            ):
                batch = _prepare_batch(batch, device)
                if batch is None:
                    continue

                with torch.no_grad():
                    loss, step_losses, step_accuracies = _rollout(
                        model,
                        batch,
                        train_diffusion_steps,
                        rollout_loss_decay,
                    )

                    reduced_loss = _distributed_mean(loss.item(), device)
                    reduced_accuracy = _distributed_mean(step_accuracies[-1], device)

                    test_loss.append(reduced_loss)
                    test_acc.append(reduced_accuracy)
                    test_step_loss.append([_distributed_mean(item.item(), device) for item in step_losses])
                    test_step_acc.append([_distributed_mean(item, device) for item in step_accuracies])

                total_loss_test += reduced_loss

            if was_training:
                model.train()

        average_loss_train = total_loss_train / len(train_loader)
        average_loss_test = total_loss_test / len(test_loader) if test_loader else 0

        if is_main_process:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train_loss={average_loss_train:.4f} | "
                f"test_loss={average_loss_test:.4f} | "
                f"train_acc={sum(train_acc) / len(train_acc):.4f} | "
                f"test_acc={sum(test_acc) / len(test_acc):.4f} | "
                f"corruption_method={str(train_loader.collate_fn.corruption_method)}"
            )

    return TrainingOutput(
        train_loss=train_loss,
        test_loss=test_loss,
        train_acc=train_acc,
        test_acc=test_acc,
        iterations_intervals=train_loader.collate_fn.corruption_method.iterations_intervals,
        train_step_loss=train_step_loss,
        test_step_loss=test_step_loss,
        train_step_acc=train_step_acc,
        test_step_acc=test_step_acc,
    )
