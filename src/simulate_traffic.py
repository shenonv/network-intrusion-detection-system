"""Step 5 demo: stream real test flows to the websocket IDS server.

Samples random flows from the held-out test set, converts them back to
raw feature values (the server applies its own scaling), sends each over
the websocket and compares the server's verdict with the true label.
Open http://127.0.0.1:8765 first to watch the dashboard react live.

Usage (server must be running first):
    python src/ws_server.py                              # terminal 1
    python src/simulate_traffic.py --flows 40 --delay 0.2  # terminal 2
"""
import argparse
import asyncio
import json

import numpy as np
import websockets

from config import RANDOM_STATE
from data_loader import load_feature_names, load_label_encoder, load_processed_data, load_scaler

SERVER_URL = "ws://127.0.0.1:8765"


async def run(n_flows, delay):
    print("Loading test flows...")
    _, X_test_scaled, _, y_test = load_processed_data()
    scaler = load_scaler()
    feature_names = load_feature_names()
    class_names = list(load_label_encoder().classes_)

    X_test_raw = scaler.inverse_transform(X_test_scaled)

    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.choice(len(X_test_raw), size=n_flows, replace=False)

    print(f"Streaming {n_flows} flows to {SERVER_URL} ...\n")
    correct = alerts = 0

    async with websockets.connect(SERVER_URL) as ws:
        for i, idx in enumerate(indices, start=1):
            flow = {name: float(v) for name, v in zip(feature_names, X_test_raw[idx])}
            true_label = class_names[y_test[idx]]

            await ws.send(json.dumps({"type": "flow", "features": flow}))
            pred = json.loads(await ws.recv())
            if pred["type"] == "error":
                print(f"flow {i}: server error: {pred['message']}")
                continue

            hit = pred["label"] == true_label
            correct += hit
            alerts += pred["is_attack"]

            marker = "ALERT " if pred["is_attack"] else "      "
            check = "correct" if hit else "WRONG  "
            print(f"{marker}flow {i:>3}: predicted={pred['label']:<24} "
                  f"p(attack)={pred['attack_probability']:.3f}  true={true_label:<24} [{check}]")

            await asyncio.sleep(delay)

    print(f"\nDone. {correct}/{n_flows} correct, {alerts} alerts raised.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows", type=int, default=30, help="number of flows to send")
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between flows")
    args = parser.parse_args()
    asyncio.run(run(args.flows, args.delay))
