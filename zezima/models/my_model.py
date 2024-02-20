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
        bp_vector_schema: list[str],
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.embedding = CustomEmbedding(
            num_embeddings=input_size,
            d_model=d_model,
            bp_vector_schema=bp_vector_schema,
        )
        self.pos_encoder = PositionalEncoding(d_model, seq_length, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_encoder_layers, enable_nested_tensor=False
        )
        self.encoder = nn.Linear(d_model, d_model)
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


class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, d_model: int, bp_vector_schema: list[str]):
        super(CustomEmbedding, self).__init__()
        self.input_size = num_embeddings
        self.d_model = d_model  # embedding dim
        self.bp_vector_schema = (
            bp_vector_schema  # Assuming this is provided during initialization
        )

        # Define embedding layers for each categorical feature
        self.embedding_layers = nn.ModuleDict(
            {
                feature: nn.Embedding(
                    num_embeddings=self.input_size, embedding_dim=self.d_model
                )
                for feature in self.bp_vector_schema
                if feature not in ["A", "C", "G", "T"]
            }
        )
        self.projection_layer = nn.Linear(self.calculate_input_dim(), self.d_model)

    def calculate_input_dim(self):
        # Calculate the dimensionality of the concatenated input
        # 4 for the binary nucleotide features + embedding_dim for each categorical feature
        num_categorical_features = (
            len(
                [
                    feature
                    for feature in self.bp_vector_schema
                    if feature not in ["A", "C", "G", "T"]
                ]
            )
            - 1
        )  # cuz gene
        return 4 + num_categorical_features * self.d_model

    def forward(self, x: torch.Tensor):
        # x is expected to be of shape [batch_size, seq_len, d_model]

        # Process binary features (nucleotides) directly; assume they are the first 4 features
        # This retains the batch and sequence dimensions
        nucleotides = x[:, :, :4]  # No embedding needed for binary features

        # Initialize a list to hold the embeddings for categorical features, starting with nucleotides
        feature_vectors = [nucleotides]

        # Start index for categorical features in the vector
        cat_feature_start_index = 4

        # Iterate over categorical features and their corresponding embedding layers
        for feature in self.bp_vector_schema[
            cat_feature_start_index : len(self.bp_vector_schema) - 1
        ]:
            # Get the indices for the current feature across the batch and sequence
            # Reshape to combine batch and sequence dimensions to apply embeddings

            feature_indices = x[:, :, cat_feature_start_index].long().view(-1)

            # Get the embedding vectors for the current feature and restore batch and sequence dimensions
            feature_vector = self.embedding_layers[feature](feature_indices)
            feature_vector = feature_vector.view(
                x.shape[0], x.shape[1], -1
            )  # Reshape to [batch_size, seq_len, embedding_dim]

            # Append the embedding vector to the list
            feature_vectors.append(feature_vector)

            # Increment the start index for the next feature
            cat_feature_start_index += 1

        # Concatenate all vectors (nucleotides + categorical feature embeddings) along the last dimension
        model_input = torch.cat(feature_vectors, dim=-1)
        model_input = self.projection_layer(model_input)
        return model_input


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
        pe = self.pe[: x.size(1), :]
        # Add positional encoding to x
        x = x + pe
        # Apply dropout
        x = self.dropout(x)
        return x
