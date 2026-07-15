"""Run the full modelling pipeline in one process:
train RF -> train XGB -> compare -> threshold analysis -> SHAP.

Usage:  python src/run_pipeline.py
"""
import traceback

import compare_models
import explain_shap
import threshold_analysis
import train_rf
import train_xgb

STEPS = [
    ("Random Forest training", train_rf.main),
    ("XGBoost training", train_xgb.main),
    ("Model comparison", compare_models.main),
    ("Threshold analysis", threshold_analysis.main),
    ("SHAP explainability", explain_shap.main),
]


def main():
    for name, step in STEPS:
        print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)
        try:
            step()
        except Exception:
            print(f"STEP_FAILED: {name}\n{traceback.format_exc()}", flush=True)
            raise SystemExit(1)
    print("\nPIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
