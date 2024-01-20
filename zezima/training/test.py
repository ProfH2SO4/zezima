import time
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from zezima import log


def validate_model(model, criterion, data_loader, state_matrix) -> None:
    model.eval()  # Set the model to evaluation mode

    total_loss: int = 0
    all_predictions: list[int] = []
    all_true_labels: list[int] = []
    # Evaluation loop
    with torch.no_grad():  # No need to track gradients for testing
        for batch_idx, batch in enumerate(data_loader):
            src, _ = batch
            batch_size, seq_len, _ = src.shape

            # Reshape or slice the state_matrix to match the src dimensions
            if state_matrix.shape[1] != seq_len:
                state_matrix = state_matrix[:, :seq_len, :]
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


def test_model(model, criterion, data_loader, state_matrix):
    model.eval()  # Set the model to evaluation mode

    total_loss: int = 0
    all_predictions: list[int] = []
    all_true_labels: list[int] = []
    start_time = time.time()

    # Evaluation loop
    with torch.no_grad():  # No need to track gradients for testing
        for batch_idx, batch in enumerate(data_loader):
            src, _ = batch
            batch_size, seq_len, _ = src.shape

            # Reshape or slice the state_matrix to match the src dimensions
            if state_matrix.shape[1] != seq_len:
                state_matrix = state_matrix[:, :seq_len, :]
            inputs, targets = batch
            output, _ = model(inputs, state_matrix.detach())
            loss = criterion(output, targets)
            total_loss += loss.item()

            softmaxed_output = torch.softmax(output, dim=-1)
            predicted_classes = torch.argmax(softmaxed_output, dim=-1).view(-1).tolist()
            true_classes = torch.argmax(targets, dim=-1).view(-1).tolist()

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
