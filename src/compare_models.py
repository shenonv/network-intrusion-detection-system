"""Step 3c: Compare all trained models side by side.

Usage:  python src/compare_models.py
Reads every reports/metrics_*.json and prints a comparison table,
also saved to reports/model_comparison.csv for the report.
"""
import json

import pandas as pd

from config import REPORTS_DIR


def main():
    metric_files = sorted(REPORTS_DIR.glob("metrics_*.json"))
    if not metric_files:
        print("No metrics found. Run train_rf.py / train_xgb.py first.")
        return

    rows = [json.loads(path.read_text()) for path in metric_files]
    table = pd.DataFrame(rows).set_index("model").round(4)

    print("\nModel comparison:")
    print(table.to_string())

    out_path = REPORTS_DIR / "model_comparison.csv"
    table.to_csv(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
