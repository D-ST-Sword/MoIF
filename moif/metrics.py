"""Original evaluation definitions; classification values are fractions."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def regression_metrics(target, prediction):
    y, p = np.asarray(target).reshape(-1), np.asarray(prediction).reshape(-1)
    assert np.isfinite(p).all()
    truth, predicted = (y >= 0).astype(int), (p >= 0).astype(int)
    return {"mae": float(np.abs(p - y).mean()),
            "corr": float(np.corrcoef(p, y)[0, 1]) if p.std() > 1e-12 else 0.,
            "accuracy": float(accuracy_score(truth, predicted)),
            "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "classification_report": classification_report(truth, predicted, labels=[0, 1], target_names=["non_humorous", "humorous"], output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(), "n_samples": len(y)}


def binary_report(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
            "classification_report": classification_report(truth, predicted, labels=[0, 1],
                target_names=["negative_side", "positive_side"], output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(), "n_samples": len(truth)}


def score(y, p):
    y, p = np.asarray(y).reshape(-1), np.asarray(p).reshape(-1)
    assert np.isfinite(y).all() and np.isfinite(p).all() and len(y) == len(p)
    nonzero = y != 0
    result = {"mae": float(np.abs(y - p).mean()), "corr": float(np.corrcoef(y, p)[0, 1]) if p.std() > 1e-12 else 0.,
              "n_samples": len(y), "has0_ge_zero": binary_report(y >= 0, p >= 0),
              "has0_gt_zero": binary_report(y > 0, p > 0),
              "non0_gt_zero": binary_report(y[nonzero] > 0, p[nonzero] > 0)}
    for classes, limit in ((7, 3), (5, 2), (3, 1)):
        truth, predicted = np.round(np.clip(y, -limit, limit)), np.round(np.clip(p, -limit, limit))
        result[f"acc_{classes}"] = float(accuracy_score(truth, predicted))
        result[f"report_{classes}"] = classification_report(truth, predicted, labels=list(range(-limit, limit + 1)),
                                                            output_dict=True, zero_division=0)
    return result


def metrics(dataset, data):
    function = regression_metrics if dataset == "urfunny" else score
    return function(data["target"], data["prediction"])
