"""MoIF neural computation, preserving the trained parameter names."""
import torch
from torch import nn
from torch.nn import functional as F
from .encoders import ModalityEncoder
from .subsets import MODALITIES, EXPERTS, SUBSETS, allowed_experts
from .text import TextEncoder
from .utils import seed_all

INPUT_DIMS = {"urfunny": (1024, 81, 371), "mosi": (1024, 5, 20), "mosei": (1024, 74, 35)}

def pool(state, mask):
    return (state * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class TextAMP(nn.Module):
    def __init__(self, encoder, amp=True):
        super().__init__()
        self.encoder = encoder
        self.amp = amp
        self.encoder.model.gradient_checkpointing_enable()

    def forward(self, tokens):
        # Keep fusion arithmetic in float32; autocast only the text backbone.
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and tokens.is_cuda):
            return self.encoder(tokens).float()


class BalancedPair(nn.Module):
    """Pool each query-aligned attention before combining unequal-length streams."""

    def __init__(self, dim=128, dropout=.1):
        super().__init__()
        self.left_norm, self.right_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.forward_attention = nn.MultiheadAttention(dim, 8, dropout=dropout, batch_first=True)
        self.reverse_attention = nn.MultiheadAttention(dim, 8, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        # Inactive compatibility tensors preserve checkpoint keys and initialization RNG.
        self.gate.requires_grad_(False)
        self.norm = nn.LayerNorm(dim)
        self.ffnorm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, left, right, left_mask, right_mask):
        a, b = self.left_norm(left), self.right_norm(right)
        x = pool(self.forward_attention(a, b, b, key_padding_mask=~right_mask,
                                       need_weights=False)[0], left_mask)
        y = pool(self.reverse_attention(b, a, a, key_padding_mask=~left_mask,
                                       need_weights=False)[0], right_mask)
        interaction = .5 * (x + y)
        residual = .5 * (pool(left, left_mask) + pool(right, right_mask))
        state = self.norm(residual + self.dropout(interaction))
        return state + self.dropout(self.ff(self.ffnorm(state)))



class SubsetPredictor(nn.Module):
    def __init__(self, dims, encoder=None, amp=True, pretrained="roberta-large"):
        super().__init__()
        if encoder is None:
            encoder = TextEncoder(pretrained)
        self.pretrained_text_encoder = TextAMP(encoder, amp)
        self.encoders = nn.ModuleDict({m: ModalityEncoder(dim, 128, .1, m) for m, dim in zip(MODALITIES, dims)})
        self.pairs = nn.ModuleDict({name: BalancedPair() for name in ("TA", "TV", "AV")})
        self.trimodal = nn.Sequential(nn.Linear(384, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.1))
        self.heads = nn.ModuleDict({name: nn.Linear(128, 1) for name in EXPERTS})

    def forward(self, batch, available=MODALITIES):
        names = allowed_experts(available)
        masks = {m: (batch["text"][:, 1].bool() if m == "text" else batch[m + "_mask"]) for m in available}
        private = {}
        for m in available:
            value = self.pretrained_text_encoder(batch[m]) if m == "text" else batch[m]
            private[m] = self.encoders[m](value, masks[m])
        states = {name: pool(private[SUBSETS[name][0]], masks[SUBSETS[name][0]]) for name in names if len(SUBSETS[name]) == 1}
        for name in ("TA", "TV", "AV"):
            if name in names:
                a, b = SUBSETS[name]
                states[name] = self.pairs[name](private[a], private[b], masks[a], masks[b])
        if "TAV" in names:
            states["TAV"] = self.trimodal(torch.cat([states[name] for name in ("TA", "TV", "AV")], -1))
        expert = torch.cat([self.heads[name](states[name]) for name in names], -1)
        return {"M": expert.mean(-1), "expert_predictions": expert, "names": names}


class MoIF(nn.Module):
    def __init__(self, dataset="urfunny", pretrained="roberta-large", encoder=None, amp=True):
        super().__init__()
        if dataset not in INPUT_DIMS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        self.core = SubsetPredictor(INPUT_DIMS[dataset], encoder, amp, pretrained)

    def forward(self, batch, available=MODALITIES):
        return self.core(batch, available)


def build(dataset, seed, pretrained="roberta-large", encoder=None, amp=True):
    seed_all(seed)
    return MoIF(dataset, pretrained, encoder, amp)


def objective(output, target):
    primary = F.l1_loss(output["M"].reshape(-1), target.reshape(-1))
    auxiliary = F.l1_loss(output["expert_predictions"], target[:, None].expand_as(output["expert_predictions"]))
    return primary + .1 * auxiliary, primary, auxiliary
