import torch


def train_model(model, criterion, optimizer, data_loader, state_matrix, num_epochs):

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
            print(f"inputs: {inputs}, targets: {targets} batch_idx: {batch_idx}")
            optimizer.zero_grad()
            output, output_state_matrix = model(inputs, state_matrix.detach())
            state_matrix = output_state_matrix
            # Compute loss
            loss = criterion(output, targets)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")

    torch.save(model.state_dict(), './saved_models/model_state.pth')