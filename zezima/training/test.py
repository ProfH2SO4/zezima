import torch


def validate_model(model, criterion, data_loader, state_matrix):
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