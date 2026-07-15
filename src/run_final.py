"""Final XGB retrain + all reports (RF artifacts already on disk)."""
import compare_models, explain_shap, threshold_analysis, train_xgb

for name, step in [("XGBoost training", train_xgb.main),
                   ("Model comparison", compare_models.main),
                   ("Threshold analysis", threshold_analysis.main),
                   ("SHAP explainability", explain_shap.main)]:
    print(f"\n==== {name} ====", flush=True)
    step()
print("\nPIPELINE_DONE", flush=True)
