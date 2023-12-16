import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):

    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers, enable_nested_tensor=False)
        self.encoder = nn.Linear(input_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 3)

    def forward(self, src, start_pos: int=0):
        src = self.pos_encoder(src, start_pos)
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, start_pos: int=0):
        pe = self.pe[start_pos:start_pos + x.size(1)].repeat(1, x.size(0), 1)
        x = x + pe
        temp = self.dropout(x)
        return temp

