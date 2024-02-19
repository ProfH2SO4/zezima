import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        seq_length: int,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_length, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_encoder_layers, enable_nested_tensor=False
        )
        self.encoder = nn.Linear(input_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 4)

        # Apply He initialization to linear layers
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, src, state_matrix):
        src = self.embedding(src)
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
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        pe = self.pe[:x.size(1), :]
        # Add positional encoding to x
        x = x + pe
        # Apply dropout
        x = self.dropout(x)
        return x
