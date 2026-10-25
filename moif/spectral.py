"""Equivalent Mobius-coordinate view; not an extra neural forward module."""
import argparse
from pathlib import Path
import numpy as np
from .subsets import EXPERTS, SUBSETS


def decompose(predictions):
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape[-1] != 7:
        raise ValueError("Expected T, A, V, TA, TV, AV, TAV in the final dimension")
    increments = np.zeros_like(values)
    sets = [set(SUBSETS[name]) for name in EXPERTS]
    for i, subset in enumerate(sets):
        for j, smaller in enumerate(sets):
            if smaller <= subset:
                increments[..., i] += (-1) ** (len(subset) - len(smaller)) * values[..., j]
    orders = np.stack([increments[..., :3].sum(-1), increments[..., 3:6].sum(-1),
                       increments[..., 6]], -1)
    filtered = orders @ (np.array([4., 2., 1.]) / 7.)
    return {"increments": increments, "orders": orders, "filtered": filtered,
            "complete_input": orders.sum(-1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="predictions.npz from moif.evaluate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        if tuple(data["expert_names"]) != EXPERTS:
            raise ValueError("This command requires a complete-input seven-prediction evaluation")
        result = decompose(data["expert_predictions"])
        np.testing.assert_allclose(result["filtered"], data["prediction"], atol=1e-6, rtol=1e-6)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **result)


if __name__ == "__main__":
    main()
