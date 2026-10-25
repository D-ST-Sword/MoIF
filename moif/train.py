"""Independent five-seed fits with validation-MAE checkpoint selection."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch
import transformers

from .data import load_data
from .evaluate import collect
from .metrics import metrics
from .model import build, objective
from .utils import seed_all, sha, to_device, write_json

SEEDS = {"urfunny": (7, 22, 42, 93, 99), "mosi": (22, 42, 93, 123, 456),
         "mosei": (22, 42, 93, 123, 456)}


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        loaders, manifest = load_data(args.data, args.dataset, args.seed, args.pretrained)
        write_json(output / "data_manifest.json", manifest)
        config = {
            "method": "MoIF", "variant": "fixed_gate", "dataset": args.dataset,
            "seed": args.seed, "batch_size": 32, "lr": 1e-5, "weight_decay": 1e-4,
            "max_epochs": 40, "minimum_epochs": 20, "patience": 20, "eta_min": 1e-7,
            "gradient_clip": 1., "text_finetuned": True, "pretrained": args.pretrained,
            "selection": "minimum validation MAE; test only after selection",
            "precision": "text-only FP16 autocast on CUDA; fusion/master weights FP32",
            "smoke": args.smoke, "python": platform.python_version(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "numpy": np.__version__,
        }
        source = Path(__file__).parent
        write_json(output / "source_sha256.json", {p.name: sha(p) for p in sorted(source.glob("*.py"))})
        model = build(args.dataset, args.seed, args.pretrained).to(args.device)
        config["parameters"] = sum(p.numel() for p in model.parameters())
        config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        write_json(output / "config.json", config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)
        scaler = torch.cuda.amp.GradScaler(init_scale=1024., enabled=torch.device(args.device).type == "cuda")
        seed_all(args.seed)
        best, selected, stale, history = float("inf"), None, 0, []
        with (output / "epochs.jsonl").open("x") as log:
            for epoch in range(1, 41):
                model.train()
                sums, count, skipped = np.zeros(3), 0, 0
                order, epoch_start = hashlib.sha256(), time.monotonic()
                for bi, batch in enumerate(loaders["train"]):
                    order.update(json.dumps(batch["id"], separators=(",", ":")).encode())
                    batch = to_device(batch, args.device)
                    optimizer.zero_grad(set_to_none=True)
                    losses = objective(model(batch), batch["target"])
                    if not all(torch.isfinite(v).all() for v in losses):
                        raise FloatingPointError(f"Non-finite loss: epoch={epoch}, batch={bi}")
                    scaler.scale(losses[0]).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
                    old_scale = scaler.get_scale()
                    if not torch.isfinite(norm) and not scaler.is_enabled():
                        raise FloatingPointError("Non-finite full-precision gradients")
                    scaler.step(optimizer)
                    scaler.update()
                    if not torch.isfinite(norm):
                        skipped += 1
                        with (output / "numerical_events.jsonl").open("a") as stream:
                            stream.write(json.dumps({"epoch": epoch, "batch": bi, "old_scale": old_scale,
                                                     "new_scale": scaler.get_scale()}) + "\n")
                        if scaler.get_scale() >= old_scale or skipped > 3 or scaler.get_scale() < 1.:
                            raise FloatingPointError("Repeated AMP overflow; failed run retained")
                    n = len(batch["target"])
                    count += n
                    sums += np.array([float(v.detach()) for v in losses]) * n
                    if args.smoke and bi == 1:
                        break
                validation = collect(model, loaders["valid"], args.device)
                measured = metrics(args.dataset, validation)
                if args.smoke:
                    write_json(output / "smoke_result.json", {
                        "status": "passed", "test_evaluated": False, "counted_as_result_seed": False,
                        "samples_trained": count, "validation": measured,
                    })
                    print("Smoke test passed; not a completed experimental run.", flush=True)
                    return
                assert count == len(loaders["train"].dataset)
                lr = optimizer.param_groups[0]["lr"]
                scheduler.step()
                if measured["mae"] < best:
                    best, selected, stale = measured["mae"], epoch, 0
                    temporary = output / "selected.tmp.pt"
                    torch.save({"model": model.state_dict(), "epoch": epoch, "validation": measured,
                                "config": config}, temporary)
                    temporary.replace(output / "selected.pt")
                    np.savez_compressed(output / "selected_validation.npz", **validation)
                else:
                    stale += 1
                row = {
                    "epoch": epoch, "loss": float(sums[0] / count), "primary": float(sums[1] / count),
                    "auxiliary": float(sums[2] / count), "validation": measured, "lr": lr,
                    "selected_epoch": selected, "stale_epochs": stale, "train_samples": count,
                    "train_order_sha256": order.hexdigest(), "amp_skipped_steps": skipped,
                    "epoch_seconds": time.monotonic() - epoch_start,
                }
                history.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(f"{args.dataset} seed{args.seed} epoch{epoch}: validMAE={measured['mae']:.6f} "
                      f"selected={selected}", flush=True)
                if epoch >= 20 and stale >= 20:
                    break
        del optimizer, scheduler, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        checkpoint = torch.load(output / "selected.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        assert checkpoint["epoch"] == min(history, key=lambda r: r["validation"]["mae"])["epoch"]
        del checkpoint
        replay = collect(model, loaders["valid"], args.device)
        with np.load(output / "selected_validation.npz") as old:
            np.testing.assert_array_equal(old["id"], replay["id"])
            error = float(np.max(np.abs(old["prediction"] - replay["prediction"])))
            if error > 1e-6:
                raise RuntimeError(f"Selected-checkpoint replay mismatch: {error}")
        test = collect(model, loaders["test"], args.device)
        (output / "test").mkdir()
        np.savez_compressed(output / "test/predictions.npz", **test)
        measured = metrics(args.dataset, test)
        write_json(output / "test/metrics.json", measured)
        write_json(output / "result.json", {
            "status": "completed", "dataset": args.dataset, "seed": args.seed,
            "epochs": len(history), "selected_epoch": selected,
            "validation_replay_max_error": error, "test": measured,
            "checkpoint_sha256": sha(output / "selected.pt"), "seconds": time.monotonic() - start,
        })
    except BaseException as error:
        write_json(output / "failure.json", {"exception": repr(error), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SEEDS), required=True)
    parser.add_argument("--data", required=True, help="Trusted processed .pkl with original splits")
    parser.add_argument("--pretrained", default="roberta-large", help="Hugging Face ID or local snapshot")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, help="A new directory; existing runs are never overwritten")
    parser.add_argument("--smoke", action="store_true", help="Two training batches plus validation, without testing")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
