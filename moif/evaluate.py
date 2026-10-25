"""Evaluate a selected MoIF checkpoint without retraining."""
import argparse
from pathlib import Path
import numpy as np
import torch

from .data import load_data
from .metrics import metrics
from .model import build
from .subsets import MODALITIES
from .utils import to_device, write_json


@torch.no_grad()
def collect(model, loader, device, available=MODALITIES):
    model.eval()
    predictions, targets, ids, experts = [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        output = model(batch, available)
        predictions.append(output["M"].reshape(-1).float().cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
        ids.extend(batch["id"])
        experts.append(output["expert_predictions"].float().cpu().numpy())
    return {"prediction": np.concatenate(predictions), "target": np.concatenate(targets),
            "id": np.asarray(ids), "expert_predictions": np.concatenate(experts),
            "expert_names": np.asarray(output["names"])}


def load_checkpoint(path, trusted=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=not trusted)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("urfunny", "mosi", "mosei"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trusted-checkpoint", action="store_true",
                        help="Allow legacy pickle metadata; use only for a checkpoint you trust")
    parser.add_argument("--pretrained", default="roberta-large")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--available", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint, args.trusted_checkpoint)
    config = checkpoint.get("config", {})
    if config.get("dataset", args.dataset) != args.dataset:
        raise ValueError("Checkpoint dataset differs from --dataset")
    seed = config.get("seed", 42)
    model = build(args.dataset, seed, args.pretrained).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    loaders, manifest = load_data(args.data, args.dataset, seed, args.pretrained, (args.split,))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = collect(model, loaders[args.split], args.device, tuple(args.available))
    np.savez_compressed(output / "predictions.npz", **data)
    result = metrics(args.dataset, data)
    write_json(output / "metrics.json", result)
    write_json(output / "data_manifest.json", manifest)
    print(result)


if __name__ == "__main__":
    main()
