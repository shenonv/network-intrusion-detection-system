"""Helpers to load the artifacts produced by preprocess.py."""
import json

import joblib
import numpy as np

from config import MODELS_DIR, PROCESSED_DATA_DIR


def load_processed_data():
    """Return X_train, X_test, y_train, y_test (already scaled/encoded)."""
    X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def load_label_encoder():
    return joblib.load(MODELS_DIR / "label_encoder.pkl")


def load_scaler():
    return joblib.load(MODELS_DIR / "scaler.pkl")


def load_feature_names():
    with open(MODELS_DIR / "feature_names.json") as f:
        return json.load(f)
