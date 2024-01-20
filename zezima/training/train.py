import time
import torch
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
            optimizer.zero_grad()
            output, output_state_matrix = model(inputs, state_matrix.detach())
            state_matrix = output_state_matrix
            # Compute loss
            loss = criterion(output, targets)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                log.debug(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(data_loader)}, Batch Loss: {loss.item()}"
                )

        average_loss = total_loss / len(data_loader)
        log.debug(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

    total_runtime = time.time() - start_time
    log.info(f"Total Inference Time: {total_runtime} seconds")

    torch.save(model.state_dict(), model_path)
