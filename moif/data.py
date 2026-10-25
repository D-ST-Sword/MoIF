"""Raw utterances plus unaligned, pre-extracted audio/video features."""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .model import INPUT_DIMS
from .utils import sha

SPLIT_SIZES = {
    "urfunny": (7614, 980, 994),
    "mosi": (1284, 229, 686),
    "mosei": (16326, 1871, 4659),
}
SPLITS = ("train", "valid", "test")


class Split(Dataset):
    def __init__(self, record, tokenizer, dataset):
        self.ids = list(map(str, record["id"]))
        self.y = np.asarray(record["regression_labels"], np.float32).reshape(-1)
        if not np.isfinite(self.y).all():
            raise ValueError("Non-finite targets")
        if dataset == "urfunny" and not set(np.unique(self.y)) <= {-1., 1.}:
            raise ValueError("UR-FUNNY requires signed targets {-1, +1}")
        if dataset != "urfunny" and (np.abs(self.y) > 3).any():
            raise ValueError("Sentiment targets must lie in [-3, 3]")
        self.tokens = tokenizer([str(s) for s in record["raw_text"]], max_length=64,
                                padding="max_length", truncation=True, return_tensors="pt")
        self.arrays = {m: np.asarray(record[m], np.float32) for m in ("audio", "vision")}
        self.lengths = {m: np.asarray(record[m + "_lengths"], np.int64) for m in self.arrays}
        if len(self.ids) != len(set(self.ids)) or len(self.ids) != len(self.y):
            raise ValueError("Duplicate IDs or inconsistent targets")
        for m, dim in zip(("audio", "vision"), INPUT_DIMS[dataset][1:]):
            a, lengths = self.arrays[m], self.lengths[m]
            if a.ndim != 3 or a.shape[0] != len(self.ids) or a.shape[2] != dim:
                raise ValueError(f"Unexpected {m} dimensions: {a.shape}")
            if not np.isfinite(a).all() or not ((lengths >= 1) & (lengths <= a.shape[1])).all():
                raise ValueError(f"Invalid {m} values or lengths")
            if dataset == "urfunny":
                padding = np.arange(a.shape[1])[None, :] >= lengths[:, None]
                if a.shape[1] != 50 or np.count_nonzero(a[padding]):
                    raise ValueError("UR-FUNNY requires the original zero-padded 50-step features")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {
            "id": self.ids[i], "target": torch.tensor(self.y[i]),
            "text": torch.stack([self.tokens["input_ids"][i], self.tokens["attention_mask"][i]]),
            **{m: torch.from_numpy(a[i]) for m, a in self.arrays.items()},
            **{m + "_mask": torch.arange(self.arrays[m].shape[1]) < lengths[i]
               for m, lengths in self.lengths.items()},
        }


def load_data(path, dataset, seed, pretrained="roberta-large", splits=SPLITS):
    # Pickle is the original dataset format; load only files from trusted providers.
    path = Path(path)
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    datasets = {s: Split(raw[s], tokenizer, dataset) for s in splits}
    del raw
    for s, ds in datasets.items():
        expected = dict(zip(SPLITS, SPLIT_SIZES[dataset]))[s]
        if len(ds) != expected:
            raise ValueError(f"{dataset}/{s}: expected {expected} examples, found {len(ds)}")
    for i, s in enumerate(splits):
        for other in splits[i + 1:]:
            if set(datasets[s].ids) & set(datasets[other].ids):
                raise ValueError(f"Overlapping IDs in {s} and {other}")
    manifest = {
        "dataset": dataset, "data_sha256": sha(path), "pretrained": str(pretrained),
        "max_text_length": 64, "normalization": "none", "splits": {},
    }
    for s, ds in datasets.items():
        tokens = torch.stack([ds.tokens["input_ids"], ds.tokens["attention_mask"]], 1)
        manifest["splits"][s] = {
            "samples": len(ds), "ids": ds.ids, "labels": ds.y.tolist(),
            "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            "audio_shape": list(ds.arrays["audio"].shape),
            "vision_shape": list(ds.arrays["vision"].shape),
        }
    loaders = {s: DataLoader(ds, batch_size=32, shuffle=s == "train", num_workers=0,
                            generator=torch.Generator().manual_seed(seed)) for s, ds in datasets.items()}
    return loaders, manifest
