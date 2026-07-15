# Network Intrusion Detection System

Machine-learning IDS trained on the CSE-CIC-IDS2018 dataset. Classifies
network flows as **Benign** or **Infiltration** in real time over a pure
websocket architecture, with a live browser dashboard.

## Project structure

```
data/raw/            CSE-CIC-IDS2018 CSV files (not in git)
data/processed/      Scaled train/test arrays (generated, not in git)
models/              Trained models + scaler + label encoder + threshold
reports/             Metrics JSONs, comparison table, figures
src/
  config.py            Shared paths and constants
  data_loader.py       Loads processed artifacts
  preprocess.py        Step 2: clean CSVs -> train/test arrays
  train_rf.py          Step 3: Random Forest
  train_xgb.py         Step 3: XGBoost
  compare_models.py    Step 3: side-by-side comparison
  evaluation.py        Shared evaluation (metrics + figures)
  threshold_analysis.py  Step 4: decision-threshold trade-off
  explain_shap.py      Step 4: SHAP explainability
  model_service.py     Shared inference: artifacts + validation + predict
  ws_server.py         Step 5: websocket IDS server (also serves the dashboard)
  dashboard.html       Step 5: live monitoring dashboard
  simulate_traffic.py  Step 5 demo: stream flows to the server
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset CSV(s) in `data/raw/`.

## Pipeline (run in order)

```
python src/preprocess.py           # 2. clean + scale + split
python src/train_rf.py             # 3a. train Random Forest
python src/train_xgb.py            # 3b. train XGBoost
python src/compare_models.py       # 3c. comparison table
python src/threshold_analysis.py   # 4a. tune the decision threshold
python src/explain_shap.py         # 4b. SHAP feature explanations
```

## Real-time detection (step 5)

Start the server, then open the dashboard:

```
python src/ws_server.py
```

- Dashboard: **http://127.0.0.1:8765** — live stat tiles, attack-probability
  chart with the tuned decision threshold, and a scrolling alert feed
- Websocket endpoint: `ws://127.0.0.1:8765` (JSON protocol)

Sensors send `{"type": "flow", "features": {<78 raw features>}}` and get a
prediction back; dashboards send `{"type": "subscribe"}` and receive a stats
snapshot plus a push event for every analysed flow.

Demo with real test flows (in a second terminal, watch the dashboard react):

```
python src/simulate_traffic.py --flows 40 --delay 0.2
```

## Results (test set)

| Model | Accuracy | Precision (attack) | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (full depth) | 0.762 | 0.64 | 0.36 | 0.46 | 0.714 |
| XGBoost (tuned, early stopping) | 0.791 | 0.90 | 0.29 | 0.44 | 0.746 |

Reference baseline: always predicting "Benign" scores 0.718 accuracy.

Infiltration is a known hard class in CSE-CIC-IDS2018: the attack tunnels
through normal-looking connections, so its flow statistics largely overlap
benign traffic. Published studies on this day of the dataset report
similar ceilings (ROC-AUC ~0.70-0.75). Accuracy above ~0.80 on this day
alone should be treated as a red flag for data leakage, not success.
To reach high headline accuracy the standard approach is to include
additional capture days (DDoS, BruteForce, Bot), whose attacks are
near-perfectly separable. See `reports/` for the threshold trade-off
analysis and SHAP explanations.
