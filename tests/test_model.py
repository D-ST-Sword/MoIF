import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from moif import build, objective
from moif.model import INPUT_DIMS
from moif.spectral import decompose
from moif.subsets import EXPERTS, SUBSETS
from moif.utils import seed_all


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257, 1024)

    def gradient_checkpointing_enable(self):
        pass


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()

    def forward(self, text):
        return self.model.embedding(text[:, 0] % 257)


def fixture(dataset="urfunny"):
    seed_all(7)
    text = torch.stack([torch.randint(4, 200, (3, 12)), torch.ones(3, 12, dtype=torch.long)], 1)
    text[0, 1, 9:] = 0
    _, audio_dim, vision_dim = INPUT_DIMS[dataset]
    return {"id": ["a", "b", "c"], "target": torch.tensor([-1., 1., -1.]),
            "text": text, "audio": torch.randn(3, 8, audio_dim), "vision": torch.randn(3, 5, vision_dim),
            "audio_mask": torch.ones(3, 8, dtype=torch.bool),
            "vision_mask": torch.ones(3, 5, dtype=torch.bool)}


class ModelTests(unittest.TestCase):
    def test_shapes_loss_and_gradients(self):
        for dataset in INPUT_DIMS:
            model = build(dataset, 42, encoder=TinyEncoder(), amp=False)
            batch = fixture(dataset)
            output = model(batch)
            self.assertEqual(output["expert_predictions"].shape, (3, 7))
            torch.testing.assert_close(output["M"], output["expert_predictions"].mean(-1), atol=0, rtol=0)
            total, primary, auxiliary = objective(output, batch["target"])
            torch.testing.assert_close(total, primary + .1 * auxiliary)
            total.backward()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertTrue(all(not p.requires_grad for pair in model.core.pairs.values()
                                for p in pair.gate.parameters()))

    def test_subset_dependencies_and_text_exclusion(self):
        model = build("urfunny", 7, encoder=TinyEncoder(), amp=False).eval()
        batch = fixture()
        with torch.no_grad():
            full = model(batch)
            for i, name in enumerate(EXPERTS):
                available = SUBSETS[name]
                restricted = {k: v for k, v in batch.items() if k not in ("text", "audio", "vision") or k in available}
                result = model(restricted, available)
                j = result["names"].index(name)
                torch.testing.assert_close(full["expert_predictions"][:, i], result["expert_predictions"][:, j])
            with patch.object(model.core.pretrained_text_encoder, "forward", side_effect=AssertionError("Text accessed")):
                model({k: v for k, v in batch.items() if k != "text"}, available=("audio", "vision"))

    def test_fixed_direction_weights(self):
        model = build("urfunny", 42, encoder=TinyEncoder(), amp=False).eval()
        pair = model.core.pairs["TA"]
        a, b = torch.randn(2, 5, 128), torch.randn(2, 3, 128)
        am, bm = torch.ones(2, 5, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool)
        with patch.object(pair.gate, "forward", side_effect=AssertionError("Inactive gate executed")):
            actual = pair(a, b, am, bm)
        x, y = pair.left_norm(a), pair.right_norm(b)
        forward = pair.forward_attention(x, y, y, need_weights=False)[0].mean(1)
        reverse = pair.reverse_attention(y, x, x, need_weights=False)[0].mean(1)
        state = pair.norm(.5 * (a.mean(1) + b.mean(1)) + .5 * (forward + reverse))
        expected = state + pair.ff(pair.ffnorm(state))
        torch.testing.assert_close(actual, expected)

    def test_mobius_identity_and_roundtrip(self):
        values = np.random.default_rng(42).normal(size=(23, 7))
        result = decompose(values)
        np.testing.assert_allclose(result["filtered"], values.mean(-1), atol=1e-14)
        np.testing.assert_allclose(result["complete_input"], values[:, 6], atol=1e-14)
        for i, name in enumerate(EXPERTS):
            indices = [j for j, other in enumerate(EXPERTS) if set(SUBSETS[other]) <= set(SUBSETS[name])]
            np.testing.assert_allclose(result["increments"][:, indices].sum(-1), values[:, i], atol=1e-14)

    def test_padding_invariance(self):
        model = build("urfunny", 42, encoder=TinyEncoder(), amp=False).eval()
        batch = fixture()
        with torch.no_grad():
            expected = model(batch)["M"]
            changed = dict(batch)
            for m in ("audio", "vision"):
                changed[m] = torch.cat([batch[m], torch.randn(3, 4, batch[m].shape[-1])], 1)
                changed[m + "_mask"] = torch.cat([batch[m + "_mask"], torch.zeros(3, 4, dtype=torch.bool)], 1)
            torch.testing.assert_close(model(changed)["M"], expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
