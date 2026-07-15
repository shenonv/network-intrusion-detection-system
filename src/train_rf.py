"""Step 3a: Train a Random Forest classifier.

Usage:  python src/train_rf.py
Output: models/random_forest.pkl + metrics/figures in reports/
"""
import time

import joblib

from sklearn.ensemble import RandomForestClassifier

from config import MODELS_DIR, RANDOM_STATE
from data_loader import load_feature_names, load_label_encoder, load_processed_data
from evaluation import evaluate_model, save_feature_importance

MODEL_NAME = "random_forest"


def main():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()
    class_names = list(load_label_encoder().classes_)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Classes: {class_names}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,            # full depth: let trees separate the hard class
        min_samples_leaf=1,
        # no class_weight: balanced weighting was measured to cost ~20 points of
        # accuracy at full depth; threshold_analysis.py handles the recall trade-off
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print("\nTraining Random Forest...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.1f} seconds.")

    model_path = MODELS_DIR / f"{MODEL_NAME}.pkl"
    joblib.dump(model, model_path, compress=3)
    print(f"Model saved: {model_path}")

    evaluate_model(model, MODEL_NAME, X_test, y_test, class_names, train_time)
    save_feature_importance(model, load_feature_names(), MODEL_NAME)


if __name__ == "__main__":
    main()
