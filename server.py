"""
server.py — FastAPI application with WebSocket hub for real-time metric streaming.

Serves the static dashboard, provides REST endpoints for training/inference/params,
and streams metrics over WebSocket at ~10Hz during training.
"""

import asyncio
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# These get set by launch.py after model loading
_trainer = None
_bio_model = None
_self_modify = None
_metric_computer = None
_config = None
_ws_clients: list[WebSocket] = []
_last_ai_metrics: dict = {}
_last_bio_metrics: dict = {}


def set_components(trainer, bio_model, self_modify_engine, metric_computer, config):
    global _trainer, _bio_model, _self_modify, _metric_computer, _config
    _trainer = trainer
    _bio_model = bio_model
    _self_modify = self_modify_engine
    _metric_computer = metric_computer
    _config = config


# --- Request models ---

class TrainRequest(BaseModel):
    text: str
    num_steps: int | None = None


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128


class ParamUpdate(BaseModel):
    name: str
    value: float


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] Neural Simulation server ready")
    yield
    print("[server] Shutting down...")
    if _trainer:
        _trainer.stop()


# --- App ---

app = FastAPI(title="Neural Simulation", lifespan=lifespan)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Routes ---

@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"error": "Dashboard not found"}, status_code=404)


@app.get("/api/status")
async def status():
    state = _trainer.get_state() if _trainer else {}
    gpu_info = {}
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch and torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024),
            "memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024),
            "memory_total_mb": round(getattr(torch.cuda.get_device_properties(0), 'total_memory', 0) / 1024 / 1024),
        }
    return {
        "training": state,
        "gpu": gpu_info,
        "self_modify_enabled": _self_modify.enabled if _self_modify else False,
        "self_modify_history_count": len(_self_modify.history) if _self_modify else 0,
    }


@app.post("/api/train")
async def start_training(req: TrainRequest):
    if not _trainer:
        raise HTTPException(503, "Trainer not initialized")
    if _trainer.state.running:
        raise HTTPException(409, "Training already running")

    # Start training in background
    asyncio.create_task(_training_loop(req.text, req.num_steps))
    return {"status": "training_started", "text_length": len(req.text)}


@app.post("/api/train/pause")
async def pause_training():
    if _trainer:
        _trainer.pause()
    return {"status": "paused"}


@app.post("/api/train/resume")
async def resume_training():
    if _trainer:
        _trainer.resume()
    return {"status": "resumed"}


@app.post("/api/train/stop")
async def stop_training():
    if _trainer:
        _trainer.stop()
    return {"status": "stopped"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not _trainer:
        raise HTTPException(503, "Model not initialized")

    result = await _trainer.infer(req.prompt, req.max_tokens)

    # Also broadcast inference metrics if we have hook data
    return {
        "text": result["text"],
        "attention_layers": len(result.get("attention_data", [])),
    }


@app.post("/api/param")
async def update_param(req: ParamUpdate):
    if not _trainer:
        raise HTTPException(503, "Trainer not initialized")
    _trainer.update_param(req.name, req.value)
    return {"status": "updated", "param": req.name, "value": req.value}


@app.post("/api/self_modify")
async def trigger_self_modify():
    """Manually trigger a self-modification evaluation cycle."""
    if not _self_modify or not _trainer:
        raise HTTPException(503, "Self-modification engine not initialized")

    # Need recent metrics
    state = _trainer.get_state()
    # Use last known metrics or compute fresh ones
    return {"status": "triggered", "note": "Will evaluate on next training step"}


@app.get("/api/self_modify/history")
async def self_modify_history():
    if not _self_modify:
        return {"history": []}
    return {"history": _self_modify.get_history()}


# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    print(f"[server] WebSocket client connected ({len(_ws_clients)} total)")

    try:
        while True:
            # Receive messages from client
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                await _handle_ws_message(ws, msg)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
    except WebSocketDisconnect:
        _ws_clients.remove(ws)
        print(f"[server] WebSocket client disconnected ({len(_ws_clients)} remaining)")
    except Exception:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


async def _handle_ws_message(ws: WebSocket, msg: dict):
    """Handle incoming WebSocket messages from the client."""
    msg_type = msg.get("type")

    if msg_type == "set_param":
        if _trainer:
            _trainer.update_param(msg["name"], msg["value"])
            await _broadcast({"type": "param_updated", "name": msg["name"], "value": msg["value"]})

    elif msg_type == "chat":
        if _trainer:
            result = await _trainer.infer(msg.get("prompt", ""), msg.get("max_tokens", 128))
            await ws.send_json({
                "type": "chat_response",
                "text": result["text"],
            })

    elif msg_type == "train_data":
        if _trainer and not _trainer.state.running:
            asyncio.create_task(_training_loop(msg.get("text", ""), msg.get("num_steps")))
            await ws.send_json({"type": "status", "message": "Training started"})

    elif msg_type == "train_control":
        action = msg.get("action")
        if _trainer:
            if action == "pause":
                _trainer.pause()
            elif action == "resume":
                _trainer.resume()
            elif action == "stop":
                _trainer.stop()
            await _broadcast({"type": "train_state", "action": action})

    elif msg_type == "request_self_modify":
        print(f"[server] Self-modify requested. Has metrics: {bool(_last_ai_metrics)}")
        if _self_modify and _trainer:
            ai = _last_ai_metrics or {}
            bio = _last_bio_metrics or {}
            if not ai:
                await _broadcast({"type": "error", "message": "No metrics yet — run at least one training step first"})
            else:
                try:
                    mod = await _self_modify.evaluate(ai, bio)
                    if mod:
                        await _broadcast({
                            "type": "self_modify_event",
                            "parameter": mod.parameter,
                            "old_value": mod.old_value,
                            "new_value": mod.new_value,
                            "reason": mod.reason,
                            "step": mod.step,
                        })
                    else:
                        await _broadcast({"type": "status", "message": "Self-modify: model found no issues to fix"})
                except Exception as e:
                    print(f"[server] Self-modify error: {e}")
                    await _broadcast({"type": "error", "message": f"Self-modify failed: {e}"})


async def _broadcast(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# --- Training loop with metric broadcasting ---

async def _training_loop(text: str, num_steps: int | None = None):
    """Run training with metric emission to WebSocket clients."""
    if not _trainer or not _bio_model:
        return

    # Set up metric callback
    last_bio_metrics = {}

    async def on_metrics(ai_metrics: dict, step: int):
        nonlocal last_bio_metrics
        global _last_ai_metrics, _last_bio_metrics

        # Run bio model step synchronized with AI
        token_ids = []
        if _trainer.tokenizer:
            enc = _trainer.tokenizer(text[:200], truncation=True, max_length=50)
            token_ids = enc["input_ids"]

        current = _bio_model.encode_text_to_current(token_ids)
        bio_m = _bio_model.simulate_step(current)
        bio_dict = _metric_computer.bio_to_dict(bio_m)
        last_bio_metrics = bio_dict
        _last_ai_metrics = ai_metrics
        _last_bio_metrics = bio_dict

        # Broadcast combined metrics
        await _broadcast({
            "type": "metrics",
            "ai": ai_metrics,
            "bio": bio_dict,
            "step": step,
            "timestamp": time.time(),
        })

        # Check self-modification
        if _self_modify and _self_modify.should_evaluate(step):
            mod = await _self_modify.evaluate(ai_metrics, bio_dict)
            if mod:
                await _broadcast({
                    "type": "self_modify_event",
                    "parameter": mod.parameter,
                    "old_value": mod.old_value,
                    "new_value": mod.new_value,
                    "reason": mod.reason,
                    "step": mod.step,
                })

    _trainer.on_metrics = on_metrics

    await _broadcast({"type": "train_state", "action": "started"})
    await _trainer.train_on_text(text, num_steps)
    await _broadcast({"type": "train_state", "action": "completed"})
