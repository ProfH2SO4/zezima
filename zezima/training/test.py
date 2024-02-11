import time
import torch
import torch.nn as nn
from torch import device
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from zezima import log
from zezima.models.my_model import TransformerModel


def validate_model(
    model: TransformerModel,
    criterion: nn.CrossEntropyLoss,
    data_loader: DataLoader,
    state_matrix: torch.Tensor,
) -> None:
    model.eval()  # Set the model to evaluation mode

    total_loss: int = 0
    all_predictions: list[int] = []
    all_true_labels: list[int] = []
    # Evaluation loop
    with torch.no_grad():  # No need to track gradients for testing
        data_loader.dataset.reset_window()
        for batch_idx, batch in enumerate(data_loader):
            inputs: torch.Tensor
            targets: torch.Tensor
            inputs, targets = batch
            batch_size, seq_len, _ = inputs.shape

            inputs, targets = batch
            output, _ = model(inputs, state_matrix.detach())

            loss = criterion(output, targets)
            total_loss += loss.item()

            predictions = torch.argmax(output, dim=-1).view(-1).tolist()
            true_labels = torch.argmax(targets, dim=-1).view(-1).tolist()

            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)

        log.debug(f" Validation Loss: {total_loss / len(data_loader)}")
    correct_predictions = sum(p == t for p, t in zip(all_predictions, all_true_labels))
    total_predictions = len(all_predictions)

    cm = confusion_matrix(all_true_labels, all_predictions)
    average_accuracy = correct_predictions / total_predictions
    precision = precision_score(
        all_true_labels, all_predictions, average="weighted", zero_division=1
    )
    recall = recall_score(
        all_true_labels, all_predictions, average="weighted", zero_division=1
    )
    f1 = f1_score(all_true_labels, all_predictions, average="weighted", zero_division=1)

    cm_str = "\n".join(["\t".join([str(cell) for cell in row]) for row in cm])
    log.info(f"Confusion Matrix: {cm_str}")
    log.info(f"Average Validation Accuracy: {average_accuracy}")
    log.info(f"Precision: {precision}")
    log.info(f"Recall: {recall}")
    log.info(f"F1 Score: {f1}")


def test_model(
    model: TransformerModel,
    criterion: nn.CrossEntropyLoss,
    data_loader: DataLoader,
    state_matrix: torch.Tensor,
    target_device: device,
) -> None:
    model.eval()  # Set the model to evaluation mode

    total_loss: int = 0
    all_predictions: list[int] = []
    all_true_labels: list[int] = []
    start_time = time.time()

    # Evaluation loop
    with torch.no_grad():  # No need to track gradients for testing
        data_loader.dataset.reset_window()
        for batch_idx, batch in enumerate(data_loader):
            inputs: torch.Tensor
            targets: torch.Tensor
            inputs, targets = batch
            batch_size, seq_len, _ = inputs.shape

            inputs, targets = inputs.to(target_device), targets.to(target_device)
            output, output_state_matrix = model(inputs, state_matrix.detach())
            state_matrix = output_state_matrix

            _, targets_indices = targets.max(dim=-1)
            output_indices = output.view(-1, 4)
            targets_indices = targets_indices.view(-1)
            # Compute loss
            loss = criterion(output_indices, targets_indices)
            total_loss += loss.item()

            predicted_classes: list[int] = output_indices.argmax(dim=-1).tolist()
            true_classes: list[int] = targets_indices.tolist()

            all_predictions.extend(predicted_classes)
            all_true_labels.extend(true_classes)

            # Optional: Batch-level logging
            if (batch_idx + 1) % 100 == 0:  # Log every 100 batches, for example
                batch_correct_predictions = sum(
                    p == t for p, t in zip(predicted_classes, true_classes)
                )
                batch_accuracy = batch_correct_predictions / len(predicted_classes)
                log.debug(
                    f"Batch {batch_idx + 1}/{len(data_loader)}, Batch Accuracy: {batch_accuracy}"
                )

        log.debug(f" Testing Loss: {total_loss / len(data_loader)}")

    # Total runtime
    total_runtime = time.time() - start_time
    log.info(f"Total Inference Time: {total_runtime} seconds")

    correct_predictions = sum(p == t for p, t in zip(all_predictions, all_true_labels))
    total_predictions = len(all_predictions)

    cm = confusion_matrix(all_true_labels, all_predictions)
    average_accuracy = correct_predictions / total_predictions
    precision = precision_score(
        all_true_labels, all_predictions, average="weighted", zero_division=1
    )
    recall = recall_score(
        all_true_labels, all_predictions, average="weighted", zero_division=1
    )
    f1 = f1_score(all_true_labels, all_predictions, average="weighted", zero_division=1)

    cm_str = "\n".join(["\t".join([str(cell) for cell in row]) for row in cm])

    log.info(f"Average Testing Loss: {total_loss / len(data_loader)}")
    log.info(f"Average Testing Accuracy: {average_accuracy}")
    log.info(f"Precision: {precision}")
    log.info(f"Recall: {recall}")
    log.info(f"F1 Score: {f1}")
    log.info(f"Confusion Matrix:\n{cm_str}")
