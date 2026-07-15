"""Step 3b: Train an XGBoost classifier.

Usage:  python src/train_xgb.py
Output: models/xgboost.pkl + metrics/figures in reports/
"""
import time

import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import MODELS_DIR, RANDOM_STATE
from data_loader import load_feature_names, load_label_encoder, load_processed_data
from evaluation import evaluate_model, save_feature_importance

MODEL_NAME = "xgboost"


def main():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()
    class_names = list(load_label_encoder().classes_)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Classes: {class_names}")

    # Hold out 10% of the training data so early stopping can pick the
    # boosting round that generalises best.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1,
        random_state=RANDOM_STATE, stratify=y_train,
    )

    # Accuracy-optimal config from the tuning experiments. No class weighting:
    # it costs accuracy, and threshold_analysis.py recovers attack recall by
    # lowering the decision threshold instead.
    model = XGBClassifier(
        n_estimators=3000,
        max_depth=12,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        tree_method="hist",
        eval_metric="mlogloss" if len(class_names) > 2 else "logloss",
        early_stopping_rounds=60,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print("\nTraining XGBoost (with early stopping)...")
    start = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - start
    print(f"Best iteration: {model.best_iteration}")
    print(f"Training completed in {train_time:.1f} seconds.")

    model_path = MODELS_DIR / f"{MODEL_NAME}.pkl"
    joblib.dump(model, model_path, compress=3)
    print(f"Model saved: {model_path}")

    evaluate_model(model, MODEL_NAME, X_test, y_test, class_names, train_time)
    save_feature_importance(model, load_feature_names(), MODEL_NAME)


if __name__ == "__main__":
    main()
