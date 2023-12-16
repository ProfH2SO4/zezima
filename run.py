import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from zezima.dataset import LimitedDataset
from zezima.model import TransformerModel
from config import BATCH_SIZE, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, sequence_length


dataset = LimitedDataset("/home/matej/git/zezima/fake_data/first_data_ann.txt")
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = TransformerModel(input_size=input_size,
                         d_model=d_model,
                         nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         dim_feedforward=dim_feedforward
                         )


# Loss function and Optimizer
criterion = nn.MSELoss()  # or nn.CrossEntropyLoss() for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function and Optimizer
criterion = nn.MSELoss()  # or nn.CrossEntropyLoss() for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        inputs, targets = batch

        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        # Calculate the start position for the current batch
        start_pos = batch_idx * batch_size * seq_len

        optimizer.zero_grad()
        output = model(inputs, start_pos)

        # Compute loss
        loss = criterion(output, targets)
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")