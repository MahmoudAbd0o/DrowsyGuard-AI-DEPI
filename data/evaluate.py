"""
Evaluation metrics for DrowsyGuard AI (Track 3).

The core functions are pure Python and operate on plain lists, so they can be
tested without pandas. ``evaluate_csv`` adds an optional pandas loader for
labeled ground-truth files.
"""
from __future__ import annotations

from collections import Counter

DEFAULT_POSITIVE = "DROWSY"


def confusion_matrix(y_true, y_pred, positive: str = DEFAULT_POSITIVE) -> dict:
    """Binary confusion matrix for ``positive`` vs. everything else."""
    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred):
        true_pos = (true == positive)
        pred_pos = (pred == positive)
        if true_pos and pred_pos:
            tp += 1
        elif not true_pos and pred_pos:
            fp += 1
        elif true_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision(cm: dict) -> float:
    denom = cm["tp"] + cm["fp"]
    return cm["tp"] / denom if denom else 0.0


def recall(cm: dict) -> float:
    denom = cm["tp"] + cm["fn"]
    return cm["tp"] / denom if denom else 0.0


def f1_score(cm: dict) -> float:
    p, r = precision(cm), recall(cm)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def false_positive_rate(cm: dict) -> float:
    denom = cm["fp"] + cm["tn"]
    return cm["fp"] / denom if denom else 0.0


def accuracy(cm: dict) -> float:
    total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
    return (cm["tp"] + cm["tn"]) / total if total else 0.0


def coverage(valid_flags) -> float:
    """Fraction of observations that were valid measurements."""
    flags = list(valid_flags)
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def detection_latencies(times, true_flags, pred_flags) -> list[float]:
    """Seconds from each event onset to the first correct detection.

    An event is a rising edge in ``true_flags``. The latency is measured to
    the first frame inside that event where ``pred_flags`` is also True.
    Events that are never detected are skipped.
    """
    times = list(times)
    true_flags = list(true_flags)
    pred_flags = list(pred_flags)
    if not (len(times) == len(true_flags) == len(pred_flags)):
        raise ValueError("times, true_flags and pred_flags must have equal length")

    latencies = []
    n = len(times)
    i = 0
    while i < n:
        is_onset = true_flags[i] and (i == 0 or not true_flags[i - 1])
        if not is_onset:
            i += 1
            continue
        j = i
        found = None
        while j < n and true_flags[j]:
            if pred_flags[j]:
                found = j
                break
            j += 1
        if found is not None:
            latencies.append(times[found] - times[i])
        i = j + 1 if j < n else n
    return latencies


def evaluate(y_true, y_pred, positive: str = DEFAULT_POSITIVE) -> dict:
    """Return the full metric bundle for a labeled sequence."""
    cm = confusion_matrix(y_true, y_pred, positive)
    return {
        **cm,
        "precision": precision(cm),
        "recall": recall(cm),
        "f1": f1_score(cm),
        "false_positive_rate": false_positive_rate(cm),
        "accuracy": accuracy(cm),
        "samples": sum(cm.values()),
    }


def state_counts(states) -> dict:
    """Count how often each driver state occurs."""
    return dict(Counter(states))


def load_labeled_csv(path: str,
                     true_col: str = "ground_truth",
                     pred_col: str = "prediction"):
    """Load two columns from a CSV with pandas and return (y_true, y_pred)."""
    import pandas as pd

    frame = pd.read_csv(path)
    for column in (true_col, pred_col):
        if column not in frame.columns:
            raise KeyError(f"column '{column}' not found in {path}")
    return frame[true_col].tolist(), frame[pred_col].tolist()


def evaluate_csv(path: str,
                 true_col: str = "ground_truth",
                 pred_col: str = "prediction",
                 positive: str = DEFAULT_POSITIVE) -> dict:
    """Convenience wrapper: load a labeled CSV and evaluate it."""
    y_true, y_pred = load_labeled_csv(path, true_col, pred_col)
    metrics = evaluate(y_true, y_pred, positive)
    metrics["path"] = path
    return metrics
