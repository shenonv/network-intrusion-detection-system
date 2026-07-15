"""Reusable model evaluation.

Both training scripts call evaluate_model() so every model is measured
the same way, and results land in reports/ for the write-up:
  reports/metrics_<model>.json          - all headline metrics
  reports/figures/cm_<model>.png        - confusion matrix
  reports/figures/importance_<model>.png- top feature importances
"""
import json

import matplotlib

matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import FIGURES_DIR, REPORTS_DIR


def evaluate_model(model, model_name, X_test, y_test, class_names, train_time=None):
    """Evaluate a fitted model on the test set, save metrics + figures."""
    print(f"\nEvaluating {model_name}...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Works for binary now and multiclass later (if more attack days are added)
    binary = len(class_names) == 2
    average = "binary" if binary else "macro"

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba[:, 1])
        if binary
        else roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
    }
    if train_time is not None:
        metrics["train_time_seconds"] = round(train_time, 2)

    print(f"\n{model_name} results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    metrics_path = REPORTS_DIR / f"metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    _save_confusion_matrix(y_test, y_pred, class_names, model_name)

    print(f"Saved: {metrics_path}")
    return metrics


def _save_confusion_matrix(y_test, y_pred, class_names, model_name):
    cm = confusion_matrix(y_test, y_pred)
    # scale the figure with the number of classes (15 labels need room)
    side = max(6, 0.8 * len(class_names) + 2)
    fig, ax = plt.subplots(figsize=(side, side * 0.9))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False,
        xticks_rotation=45,
    )
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig_path = FIGURES_DIR / f"cm_{model_name}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def save_feature_importance(model, feature_names, model_name, top_n=20):
    """Bar chart of the top_n most important features (tree models)."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        importances[order][::-1],
        color="#1f77b4",
    )
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig_path = FIGURES_DIR / f"importance_{model_name}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")
