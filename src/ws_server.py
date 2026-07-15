"""Step 5: Real-time Network Intrusion Detection over pure websockets.

One server, one port:
  http://127.0.0.1:8765            - serves the live dashboard page
  ws://127.0.0.1:8765              - the websocket endpoint

Protocol (JSON messages):
  Sensors / simulators send:
    {"type": "flow", "features": {<78 raw feature values>}}
  and receive back:
    {"type": "prediction", "label", "is_attack", "attack_probability", "threshold"}

  Dashboards send:
    {"type": "subscribe"}
  and receive:
    {"type": "snapshot", ...}   once, with current stats
    {"type": "flow_event", ...} for every analysed flow (attack or benign)

Run:  python src/ws_server.py
Demo: python src/simulate_traffic.py --flows 40 --delay 0.2
"""
import asyncio
import json
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path

import websockets

from model_service import ModelService

HOST = "127.0.0.1"
PORT = 8765
DASHBOARD_FILE = Path(__file__).parent / "dashboard.html"

service: ModelService | None = None
subscribers: set = set()
stats = {"flows_total": 0, "attacks_total": 0}


def _now():
    return datetime.now(timezone.utc).isoformat()


async def broadcast(message: dict):
    """Push a message to every dashboard; drop clients that vanished."""
    if not subscribers:
        return
    data = json.dumps(message)
    results = await asyncio.gather(
        *(client.send(data) for client in list(subscribers)),
        return_exceptions=True,
    )
    for client, result in zip(list(subscribers), results):
        if isinstance(result, Exception):
            subscribers.discard(client)


async def handle_flow(websocket, message):
    features = message.get("features")
    if not isinstance(features, dict):
        await websocket.send(json.dumps(
            {"type": "error", "message": "'features' must be an object of {name: value}"}))
        return

    try:
        prediction = service.predict_one(features)
    except ValueError as exc:
        await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
        return

    stats["flows_total"] += 1
    stats["attacks_total"] += prediction["is_attack"]

    # Reply to the sensor that sent the flow
    await websocket.send(json.dumps({"type": "prediction", **prediction}))

    # Push to every dashboard
    await broadcast({
        "type": "flow_event",
        "timestamp": _now(),
        "flow_id": stats["flows_total"],
        **prediction,
    })


async def handle_subscribe(websocket):
    subscribers.add(websocket)
    await websocket.send(json.dumps({
        "type": "snapshot",
        "timestamp": _now(),
        "classes": service.classes,
        "threshold": service.threshold,
        **stats,
    }))


async def handler(websocket):
    try:
        async for raw in websocket:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(json.dumps(
                    {"type": "error", "message": "invalid JSON"}))
                continue

            kind = message.get("type")
            if kind == "flow":
                await handle_flow(websocket, message)
            elif kind == "subscribe":
                await handle_subscribe(websocket)
            else:
                await websocket.send(json.dumps(
                    {"type": "error", "message": f"unknown message type: {kind!r}"}))
    finally:
        subscribers.discard(websocket)


def process_request(connection, request):
    """Serve the dashboard to plain HTTP requests; let websockets through."""
    if "upgrade" in request.headers.get("Connection", "").lower():
        return None  # continue with the websocket handshake
    if request.path == "/":
        response = connection.respond(HTTPStatus.OK, DASHBOARD_FILE.read_text(encoding="utf-8"))
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        return response
    return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")


async def main():
    global service
    print("Loading model artifacts...")
    service = ModelService()
    print(f"Model ready: {service.classes} @ threshold {service.threshold}")
    async with websockets.serve(handler, HOST, PORT, process_request=process_request):
        print(f"IDS server running:")
        print(f"  dashboard  http://{HOST}:{PORT}")
        print(f"  websocket  ws://{HOST}:{PORT}")
        await asyncio.Future()  # run until cancelled


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
