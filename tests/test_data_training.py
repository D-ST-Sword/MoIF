import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from moif.data import Split
from moif.metrics import regression_metrics, score
from moif.train import train


class TokenizerFixture:
    def __call__(self, text, **kwargs):
        n = len(text)
        assert kwargs == {"max_length": 64, "padding": "max_length", "truncation": True, "return_tensors": "pt"}
        return {"input_ids": torch.arange(64).repeat(n, 1),
                "attention_mask": torch.ones(n, 64, dtype=torch.long)}


def records():
    return {"id": ["a", "b", "c"], "raw_text": ["first", "second", "third"],
            "regression_labels": [-1., 1., -1.], "audio": np.zeros((3, 50, 81)),
            "vision": np.zeros((3, 50, 371)), "audio_lengths": [7, 13, 5],
            "vision_lengths": [9, 5, 17]}


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 7)

    def forward(self, batch, available=None):
        predictions = self.head(batch["target"][:, None])
        return {"M": predictions.mean(-1), "expert_predictions": predictions,
                "names": ["T", "A", "V", "TA", "TV", "AV", "TAV"]}


class DataTrainingTests(unittest.TestCase):
    def test_input_masks_and_targets(self):
        ds = Split(records(), TokenizerFixture(), "urfunny")
        self.assertEqual(ds[0]["text"].shape, (2, 64))
        self.assertEqual(ds[1]["audio_mask"].sum(), 13)
        self.assertEqual(ds[2]["vision_mask"].sum(), 17)
        self.assertEqual(float(ds[0]["target"]), -1.)

    def test_reject_bad_features(self):
        bad = records()
        bad["audio"][0, -1, 0] = 1
        with self.assertRaises(ValueError):
            Split(bad, TokenizerFixture(), "urfunny")
        bad = records()
        bad["regression_labels"] = [0., 1., 0.]
        with self.assertRaises(ValueError):
            Split(bad, TokenizerFixture(), "urfunny")

    def test_metrics(self):
        measured = regression_metrics([-1, 1, 1, -1], [-.8, .5, 0, .2])
        self.assertEqual(measured["accuracy"], .75)
        self.assertAlmostEqual(measured["mae"], .725)
        sentiment = score([-1., 0., 1.], [-1., 0., 1.])
        self.assertEqual(sentiment["has0_ge_zero"]["accuracy"], 1.)
        self.assertEqual(sentiment["non0_gt_zero"]["n_samples"], 2)
        self.assertEqual(sentiment["acc_7"], 1.)

    def test_training_selection_and_no_smoke_test_evaluation(self):
        ds = Split(records(), TokenizerFixture(), "urfunny")
        loader = DataLoader(ds, batch_size=2)
        for smoke in [True, False]:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                args = SimpleNamespace(output=str(output), dataset="urfunny", data="unused.pkl",
                                       seed=42, pretrained="fixture", device="cpu", smoke=smoke)
                with patch("moif.train.load_data", return_value=({s: loader for s in ["train", "valid", "test"]}, {})), \
                     patch("moif.train.build", return_value=ToyModel()), \
                     patch("moif.train.range", return_value=range(1, 4), create=True):
                    train(args)
                if smoke:
                    self.assertFalse((output / "test").exists())
                    self.assertFalse((output / "selected.pt").exists())
                else:
                    rows = [json.loads(s) for s in (output / "epochs.jsonl").read_text().splitlines()]
                    result = json.loads((output / "result.json").read_text())
                    self.assertEqual(result["selected_epoch"], min(rows, key=lambda x: x["validation"]["mae"])["epoch"])
                    self.assertEqual(result["validation_replay_max_error"], 0)
                    self.assertTrue((output / "test/metrics.json").exists())
                with self.assertRaises(FileExistsError):
                    train(args)


if __name__ == "__main__":
    unittest.main()
