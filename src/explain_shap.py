"""Step 4b: SHAP explainability for the best model (XGBoost).

Answers "WHY does the model flag a flow as an attack?" - required for a
defensible IDS and great material for the report.

Uses a random sample of the test set (SHAP on all 65k rows is slow and
the plots look identical).

Usage:  python src/explain_shap.py
Output: reports/figures/shap_beeswarm_xgboost.png  - direction + strength
        reports/figures/shap_bar_xgboost.png       - global importance
        reports/shap_top_features.json             - top features, for the report text
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from data_loader import load_feature_names, load_label_encoder, load_processed_data

MODEL_NAME = "xgboost"
SAMPLE_SIZE = 1000  # the final model is large; SHAP cost grows with tree size
TOP_N = 20


def to_class_list(shap_values, n_classes):
    """Normalise SHAP output to a list of (n_samples, n_features) arrays,
    one per class. Binary models yield a single-element list.
    Handles every layout shap/xgboost versions produce: 2-D array,
    list of 2-D arrays, or 3-D array with the class axis first or last."""
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]
    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3 and arr.shape[-1] == n_classes:
        return [arr[..., k] for k in range(n_classes)]
    if arr.ndim == 3 and arr.shape[0] == n_classes:
        return list(arr)
    raise ValueError(f"unexpected SHAP output shape: {arr.shape}")


def main():
    print("Loading model and test data...")
    model = joblib.load(MODELS_DIR / f"{MODEL_NAME}.pkl")
    _, X_test, _, _ = load_processed_data()
    feature_names = load_feature_names()
    class_names = list(load_label_encoder().classes_)

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
    X_sample = X_test[idx]
    print(f"Computing SHAP values on {len(X_sample)} sampled test flows...")

    explainer = shap.TreeExplainer(model)
    per_class = to_class_list(explainer.shap_values(X_sample), len(class_names))

    if len(per_class) == 1:
        # Binary: beeswarm shows direction (right of centre -> "attack")
        shap.summary_plot(per_class[0], X_sample, feature_names=feature_names,
                          max_display=TOP_N, show=False)
        plt.title(f"SHAP Summary - {MODEL_NAME}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_beeswarm_{MODEL_NAME}.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"Saved: reports/figures/shap_beeswarm_{MODEL_NAME}.png")
    else:
        print(f"{len(per_class)} classes detected - skipping the beeswarm "
              "(only meaningful for binary models).")

    # Global importance: mean |SHAP| aggregated over samples (and classes)
    mean_abs = np.mean([np.abs(v).mean(axis=0) for v in per_class], axis=0)
    order = np.argsort(mean_abs)[::-1][:TOP_N]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([feature_names[i] for i in order][::-1],
            mean_abs[order][::-1], color="#1f77b4")
    ax.set_title(f"Mean |SHAP| Feature Importance - {MODEL_NAME}")
    ax.set_xlabel("mean |SHAP| (averaged over classes)" if len(per_class) > 1
                  else "mean |SHAP|")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"shap_bar_{MODEL_NAME}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: reports/figures/shap_bar_{MODEL_NAME}.png")

    top_features = [
        {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 5)}
        for i in order
    ]

    with open(REPORTS_DIR / "shap_top_features.json", "w") as f:
        json.dump({"model": MODEL_NAME, "sample_size": len(X_sample),
                   "top_features": top_features}, f, indent=4)

    print(f"\nTop {TOP_N} features by mean |SHAP|:")
    for rank, item in enumerate(top_features, start=1):
        print(f"  {rank:>2}. {item['feature']:<25} {item['mean_abs_shap']}")
    print("\nSaved: reports/shap_top_features.json")


if __name__ == "__main__":
    main()
