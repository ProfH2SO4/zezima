import time
import torch
import torch.nn.utils as torch_utils
from torch import Tensor

from zezima import log
from zezima.models import TransformerModel
from torch.utils.data import DataLoader


def train_model(
    model: TransformerModel,
    criterion,
    optimizer,
    data_loader: DataLoader,
    state_matrix: Tensor,
    num_epochs: int,
    model_path: str,
    device,
    max_grad_norm: float = 1.0,
):
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(data_loader):
            src, _ = batch
            batch_size, seq_len, _ = src.shape

            # Reshape or slice the state_matrix to match the src dimensions
            if state_matrix.shape[1] != seq_len:
                state_matrix = state_matrix[:, :seq_len, :]
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output, output_state_matrix = model(inputs, state_matrix.detach())
            state_matrix = output_state_matrix
            # Compute loss
            loss = criterion(output, targets)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()

            torch_utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if torch.isnan(loss).any():
                print("Loss is NaN")
            formatted_loss = f"{loss.item():.5f}"

            if (batch_idx + 1) % 100 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # Log the gradient; you can use any logging method here
                        log.debug(f"Gradient of {name}: {param.grad}")
                log.debug(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(data_loader)}, Batch Loss: {formatted_loss}"
                )

        average_loss = total_loss / len(data_loader)
        log.debug(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

    total_runtime = time.time() - start_time
    log.info(f"Total Inference Time: {total_runtime} seconds")

    torch.save(model.state_dict(), model_path)
