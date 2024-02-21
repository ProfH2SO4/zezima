import time
import torch
import torch.nn as nn
import torch.nn.utils as torch_utils
import torch.optim as optim
from torch import Tensor, device
from torch.utils.data import DataLoader

from zezima import log
from zezima.models import TransformerModel, MultiClassFocalLoss

from zezima.common import create_file_if_not_exists


def save_checkpoint(
    model: TransformerModel,
    optimizer: optim.Adam,
    loss,
    epoch: int,
    path_to_checkpoint: str,
) -> None:
    create_file_if_not_exists(path_to_checkpoint)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path_to_checkpoint,
    )


def load_checkpoint_file(path_to_checkpoint: str) -> dict | None:
    checkpoint: dict | None = None
    try:
        checkpoint: dict = torch.load(path_to_checkpoint)
        log.info("checkpoint found")
    except FileNotFoundError as e:
        log.info("NO checkpoint found")
    return checkpoint


def restore_checkpoint(checkpoint, model: TransformerModel, optimizer: optim.Adam):
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)  # Default to 0 if not found
    return epoch  # Return epoch and loss to use outside


def train_model(
    model: TransformerModel,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    data_loader: DataLoader,
    state_matrix: Tensor,
    num_epochs: int,
    model_path: str,
    target_device: device,
    path_to_checkpoint: str,
    use_checkpoint: bool,
    debug_level: int,
    max_grad_norm: float = 1.0,
):
    start_time = time.time()

    start_epoch: int = 0

    if use_checkpoint:
        checkpoint: dict | None = load_checkpoint_file(path_to_checkpoint)
        if checkpoint:
            start_epoch = restore_checkpoint(checkpoint, model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss: float = 0.0
        state_matrix_loc = state_matrix
        data_loader.dataset.reset_window()
        logging_interval = max(1, len(data_loader) // 10)
        for batch_idx, batch in enumerate(data_loader):
            inputs: torch.Tensor
            targets: torch.Tensor
            inputs, targets = batch
            batch_size, seq_len, _ = inputs.shape

            inputs, targets = inputs.to(target_device), targets.to(target_device)
            optimizer.zero_grad()
            output, output_state_matrix = model(inputs, state_matrix_loc.detach())
            state_matrix_loc = output_state_matrix

            _, targets_indices = targets.max(dim=-1)
            output_indices = output.view(-1, 2)
            targets_indices = targets_indices.view(-1)
            # Compute loss
            loss = criterion(output_indices, targets_indices)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()

            torch_utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if debug_level == 2:
                log.debug(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(data_loader)}, batch_loss: {loss.item():.5f}"
                )
            elif (batch_idx + 1) % logging_interval == 0:
                log.debug(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(data_loader)}, batch_loss: {loss.item():.5f}"
                )
        if use_checkpoint:
            save_checkpoint(model, optimizer, total_loss, epoch, path_to_checkpoint)
        average_loss = total_loss / len(data_loader)
        log.debug(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.5f}")

    total_runtime = time.time() - start_time
    log.info(f"Total Inference Time: {total_runtime} seconds")

    torch.save(model.state_dict(), model_path)
