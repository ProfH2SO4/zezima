import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def validate_model(model, criterion, data_loader, state_matrix) -> None:
    model.eval()  # Set the model to evaluation mode


    total_loss: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0
    all_predictions: list[list[int]] = []
    all_true_labels: list[list[int]] = []
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

            predictions = torch.argmax(output, dim=-1)
            true_labels = torch.argmax(targets, dim=-1)

            all_predictions.extend(predictions.tolist())
            all_true_labels.extend(true_labels.tolist())

            # Compare predictions with true labels
            correct_predictions += (predictions == true_labels).sum().item()
            total_predictions += true_labels.numel()

        print(f" Validation Loss: {total_loss / len(data_loader)}")
    average_accuracy = correct_predictions / total_predictions
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    print(f'Average Validation Accuracy: {average_accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


def test_model(model, criterion, data_loader, state_matrix):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    # Evaluation loop
    total_accuracy = 0
    with torch.no_grad():  # No need to track gradients for testing
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            output, _ = model(inputs, state_matrix.detach())
            loss = criterion(output, targets)
            total_loss += loss.item()

        print(f" Testing Loss: {total_loss / len(data_loader)}")
    average_accuracy = total_accuracy / len(data_loader)
    print(f'Average Test Accuracy: {average_accuracy}')