"""Private temporal encoders used in the reported runs."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def mask_states(x, mask):
    return x.masked_fill(~mask.unsqueeze(-1), 0)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, maximum=2048):
        super().__init__()
        position = torch.arange(maximum, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        values = torch.zeros(maximum, dim)
        values[:, 0::2], values[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        self.register_buffer("values", values, persistent=False)

    def forward(self, x):
        if x.size(1) > len(self.values):
            raise ValueError("Dialogue exceeds positional encoding capacity")
        return x + self.values[:x.size(1)]


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, dim, dropout, kind):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.kind = kind
        if kind == "text":
            self.positions = PositionalEncoding(dim)
            layer = nn.TransformerEncoderLayer(dim, 4, 2 * dim, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        else:
            self.context = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        x = mask_states(self.projection(x), mask)
        if self.kind == "text":
            output = self.context(self.positions(x), src_key_padding_mask=~mask)
        else:
            packed = pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.context(packed)
            output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.size(1))
        return mask_states(self.output_norm(output + x), mask)
