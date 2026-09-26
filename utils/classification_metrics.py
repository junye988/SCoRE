"""Classification metrics computed from sample-level model predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def softmax_probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = np.exp(values - values.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def compute_classification_metrics(labels, scores, *, probability_input=False):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores)
    if scores.ndim != 2 or len(labels) != len(scores) or not len(labels):
        raise ValueError("Expected nonempty labels [N] and scores [N, K]")
    if not np.isfinite(scores).all():
        raise ValueError("Predictions contain nonfinite values")
    prob = scores.astype(np.float64) if probability_input else softmax_probabilities(scores)
    if probability_input and (np.any(prob < 0) or not np.allclose(prob.sum(1), 1, atol=1e-5)):
        raise ValueError("Probability rows must be nonnegative and sum to one")
    classes = np.arange(scores.shape[1])
    predicted = scores.argmax(axis=1)
    result = {
        "samples": int(len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "cohen_kappa": float(cohen_kappa_score(labels, predicted)),
        "weighted_f1": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(labels, prob, labels=classes)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=classes).tolist(),
    }
    if len(classes) == 2 and len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, prob[:, 1]))
        result["auc_pr"] = float(average_precision_score(labels, prob[:, 1]))
    return result
