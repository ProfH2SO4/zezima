import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        seq_length,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, seq_length, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_encoder_layers, enable_nested_tensor=False
        )
        self.encoder = nn.Linear(input_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 3)

    def forward(self, src, state_matrix):
        src = self.pos_encoder(src)
        src = self.encoder(src) * math.sqrt(self.d_model)

        # Add the state matrix to the source matrix
        src = src + state_matrix

        output = self.transformer_encoder(src)

        # Add the state matrix to the output
        new_state_matrix = output + state_matrix

        output = self.decoder(output)

        return output, new_state_matrix


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Reshape x for positional encoding addition
        x_expanded = x.view(-1, d_model).unsqueeze(
            1
        )  # Reshape to (batch_size * seq_len, 1, d_model)

        # Extract and expand positional encodings based on the sequence length of x
        pe = self.pe[:seq_len, :].expand_as(x_expanded)

        # Add positional encoding to x_expanded
        x_expanded = x_expanded + pe

        # Apply dropout
        x_expanded = self.dropout(x_expanded)

        # Reshape back to original x shape
        x = x_expanded.view(batch_size, seq_len, d_model)

        return x
