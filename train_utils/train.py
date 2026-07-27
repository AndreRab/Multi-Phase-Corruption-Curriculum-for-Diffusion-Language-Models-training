from dataclasses import dataclass
import torch
from tqdm import tqdm

from model_wrappers.gpt2_diffusion_transformer_wrapper import GPT2DiffusionTransformer

@dataclass
class TrainingOutput:
    train_loss: list[float]
    test_loss: list[float]
    train_acc: list[float]
    test_acc: list[float]
    iterations_intervals: dict

def train(
    device: torch.device, 
    model: GPT2DiffusionTransformer, 
    optimizer: torch.optim.Optimizer, 
    num_epochs: int, 
    train_loader: torch.utils.data.DataLoader, 
    test_loader: torch.utils.data.DataLoader = None
) -> TrainingOutput:
    
    model = model.to(device)
    model.train()

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(num_epochs):
        corruption_method = getattr(train_loader.collate_fn, "corruption_method", None)    
        corruption_method.set_epoch(epoch)

        model.train()
        total_loss_train = 0.0
        total_loss_test = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for batch in progress_bar:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            current_batch_attention_mask = batch["attention_mask"]
            valid_samples_in_batch = (
                current_batch_attention_mask.sum(dim=1) > 0
            ).nonzero(as_tuple=True)[0]

            if len(valid_samples_in_batch) == 0:
                continue

            for key in batch:
                batch[key] = batch[key][valid_samples_in_batch]

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                corrupted_ids=batch["corrupted_ids"],
                timesteps=batch["timesteps"],
                attention_mask=batch["attention_mask"],
            )

            loss = model.compute_loss(
                logits=logits,
                clean_ids=batch["clean_ids"],
                corrupted_positions=batch["corrupted_positions"],
                attention_mask=batch["attention_mask"],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            accuracy = model.compute_accuracy(
                logits=logits,
                clean_ids=batch["clean_ids"],
                corrupted_positions=batch["corrupted_positions"],
                attention_mask=batch["attention_mask"],
            )

            train_loss.append(loss.item())
            train_acc.append(accuracy)

            total_loss_train += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{accuracy:.4f}",
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
            ):
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                current_batch_attention_mask = batch["attention_mask"]
                valid_samples_in_batch = (
                    current_batch_attention_mask.sum(dim=1) > 0
                ).nonzero(as_tuple=True)[0]

                if len(valid_samples_in_batch) == 0:
                    continue

                for key in batch:
                    batch[key] = batch[key][valid_samples_in_batch]

                with torch.no_grad():
                    logits = model(
                        corrupted_ids=batch["corrupted_ids"],
                        timesteps=batch["timesteps"],
                        attention_mask=batch["attention_mask"],
                    )

                    loss = model.compute_loss(
                        logits=logits,
                        clean_ids=batch["clean_ids"],
                        corrupted_positions=batch["corrupted_positions"],
                        attention_mask=batch["attention_mask"],
                    )

                    accuracy = model.compute_accuracy(
                        logits=logits,
                        clean_ids=batch["clean_ids"],
                        corrupted_positions=batch["corrupted_positions"],
                        attention_mask=batch["attention_mask"],
                    )

                    test_loss.append(loss.item())
                    test_acc.append(accuracy)

                total_loss_test += loss.item()

            if was_training:
                model.train()

        average_loss_train = total_loss_train / len(train_loader)
        average_loss_test = total_loss_test / len(test_loader) if test_loader else 0

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
    )
