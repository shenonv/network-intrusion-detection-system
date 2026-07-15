"""Step 4a: Decision threshold analysis for the best model (XGBoost).

A classifier outputs a probability; the default 0.5 cut-off is arbitrary.
For an IDS, a missed attack (false negative) is usually more costly than
a false alarm, so we sweep the threshold and show the trade-off.

Works for binary and multiclass labels alike: the alert decision is made
on the "attack score" = 1 - P(Benign), i.e. benign-vs-any-attack.

Usage:  python src/threshold_analysis.py
Output: models/decision_threshold.json      - recommended threshold
        reports/threshold_analysis.json     - full table for the report
        reports/figures/pr_curve_*.png
        reports/figures/roc_curve_*.png
        reports/figures/threshold_sweep_*.png
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from data_loader import load_label_encoder, load_processed_data

MODEL_NAME = "xgboost"
RECALL_TARGETS = [0.70, 0.80, 0.90]  # "catch at least X% of attacks"


def metrics_at(threshold, y_test, y_proba):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "false_alarms": int(((y_pred == 1) & (y_test == 0)).sum()),
        "missed_attacks": int(((y_pred == 0) & (y_test == 1)).sum()),
    }


def main():
    print("Loading model and test data...")
    model = joblib.load(MODELS_DIR / f"{MODEL_NAME}.pkl")
    _, X_test, _, y_test = load_processed_data()
    class_names = list(load_label_encoder().classes_)

    # Benign-vs-any-attack framing (identical to the old behaviour when
    # there are only two classes)
    benign_idx = class_names.index("Benign") if "Benign" in class_names else 0
    attack_name = (class_names[1 - benign_idx] if len(class_names) == 2
                   else "any attack")

    y_proba = 1.0 - model.predict_proba(X_test)[:, benign_idx]
    y_test = (y_test != benign_idx).astype(int)

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recalls, precisions)

    # Best-F1 threshold from the PR curve
    f1s = 2 * precisions[:-1] * recalls[:-1] / np.clip(precisions[:-1] + recalls[:-1], 1e-9, None)
    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])

    rows = [
        {"scenario": "default_0.5", **metrics_at(0.5, y_test, y_proba)},
        {"scenario": "best_f1", **metrics_at(best_threshold, y_test, y_proba)},
    ]

    # Lowest threshold that still reaches each recall target
    for target in RECALL_TARGETS:
        ok = recalls[:-1] >= target
        if ok.any():
            t = float(thresholds[np.where(ok)[0][-1]])
            rows.append({"scenario": f"recall_{int(target * 100)}pct", **metrics_at(t, y_test, y_proba)})

    print(f"\nThreshold trade-off for detecting '{attack_name}' "
          f"(test set: {int((y_test == 1).sum())} attacks, {int((y_test == 0).sum())} benign):\n")
    header = f"{'scenario':<15}{'thresh':>8}{'prec':>8}{'recall':>8}{'f1':>8}{'false alarms':>14}{'missed':>8}"
    print(header)
    for r in rows:
        print(f"{r['scenario']:<15}{r['threshold']:>8}{r['precision']:>8}"
              f"{r['recall']:>8}{r['f1']:>8}{r['false_alarms']:>14}{r['missed_attacks']:>8}")

    with open(REPORTS_DIR / "threshold_analysis.json", "w") as f:
        json.dump({"model": MODEL_NAME, "pr_auc": round(pr_auc, 4), "scenarios": rows}, f, indent=4)

    with open(MODELS_DIR / "decision_threshold.json", "w") as f:
        json.dump({"model": MODEL_NAME, "threshold": round(best_threshold, 4), "chosen_by": "best_f1"}, f, indent=4)

    _plot_pr_curve(precisions, recalls, pr_auc, rows)
    _plot_roc_curve(y_test, y_proba)
    _plot_threshold_sweep(y_test, y_proba, best_threshold)

    print(f"\nRecommended threshold (best F1): {best_threshold:.4f}")
    print("Saved: models/decision_threshold.json, reports/threshold_analysis.json")


def _plot_pr_curve(precisions, recalls, pr_auc, rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, lw=2)
    for r in rows:
        ax.scatter(r["recall"], r["precision"], zorder=3)
        ax.annotate(r["scenario"], (r["recall"], r["precision"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Recall (attacks caught)")
    ax.set_ylabel("Precision (alerts that are real)")
    ax.set_title(f"Precision-Recall Curve - {MODEL_NAME} (PR-AUC = {pr_auc:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"pr_curve_{MODEL_NAME}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: reports/figures/pr_curve_{MODEL_NAME}.png")


def _plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {MODEL_NAME}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"roc_curve_{MODEL_NAME}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: reports/figures/roc_curve_{MODEL_NAME}.png")


def _plot_threshold_sweep(y_test, y_proba, best_threshold):
    sweep = np.linspace(0.05, 0.95, 91)
    results = [metrics_at(t, y_test, y_proba) for t in sweep]
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in ("precision", "recall", "f1"):
        ax.plot(sweep, [r[key] for r in results], lw=2, label=key)
    ax.axvline(best_threshold, color="gray", ls="--", lw=1,
               label=f"best F1 @ {best_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Metrics vs Threshold - {MODEL_NAME}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"threshold_sweep_{MODEL_NAME}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: reports/figures/threshold_sweep_{MODEL_NAME}.png")


if __name__ == "__main__":
    main()
