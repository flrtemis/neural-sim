"""
launch.py — Entry point for the Neural Simulation dashboard.

Loads the model, initializes all components, and starts the FastAPI server.

    cd /mnt/c/Users/l3ung/Desktop/ReverseEngineering/toolbox/neural-sim
    python launch.py [--model /path/to/model] [--port 8765] [--config config.yaml]
"""

import sys
import os
import argparse
import importlib
import platform
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))


def require_module(name: str, purpose: str, install_hint: str):
    """Import a module with an explicit actionable failure message."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency '{name}' required to {purpose}. Install it with: {install_hint}"
        ) from exc


def running_in_wsl() -> bool:
    """Return True when running under WSL/Linux where UNC paths need conversion."""
    if sys.platform != "linux":
        return False
    release = platform.release().lower()
    return "microsoft" in release or "wsl" in release or bool(os.getenv("WSL_DISTRO_NAME"))


def normalize_model_paths(config: dict) -> None:
    """Convert Windows WSL UNC paths only when actually running inside WSL."""
    if not running_in_wsl():
        return

    import re

    for key in ("path", "alt_path"):
        raw_path = config["model"].get(key, "")
        match = re.match(r'^//wsl\$/[^/]+(.+)$', raw_path)
        if match:
            config["model"][key] = match.group(1)


def resolve_model_candidates(config: dict) -> list[Path]:
    candidates: list[Path] = []
    for key in ("path", "alt_path"):
        raw_path = config["model"].get(key)
        if raw_path:
            candidates.append(Path(raw_path))
    return candidates


def require_existing_model_path(config: dict) -> Path:
    """Verify at least one configured model path exists before model load."""
    candidates = resolve_model_candidates(config)
    for candidate in candidates:
        if candidate.exists():
            config["model"]["path"] = str(candidate)
            return candidate

    joined = ", ".join(str(path) for path in candidates) if candidates else "<none configured>"
    raise RuntimeError(
        f"No configured model path exists. Checked: {joined}. "
        "Use --model to point at a valid local Hugging Face model directory or run with --no-model."
    )


def main():
    parser = argparse.ArgumentParser(description="Neural Simulation — Real-Time LLM Training Dashboard")
    parser.add_argument("--model", type=str, default=None, help="Path to HuggingFace model directory")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 8765)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--no-model", action="store_true", help="Start without loading a model (dashboard only)")
    args = parser.parse_args()

    yaml = require_module("yaml", "load the Neural Sim configuration", "pip install pyyaml")
    uvicorn = require_module("uvicorn", "serve the Neural Sim dashboard", "pip install uvicorn")

    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override from CLI
    if args.model:
        config["model"]["path"] = args.model
    if args.port:
        config["server"]["port"] = args.port

    port = config["server"]["port"]
    host = config["server"]["host"]

    normalize_model_paths(config)

    torch = None
    try:
        torch = require_module("torch", "report GPU status and load the model", "pip install torch")
    except RuntimeError as exc:
        if not args.no_model:
            raise
        print(f"  Torch: unavailable ({exc})")

    # Banner
    print("=" * 60)
    print("  NEURAL SIMULATION")
    print("  Real-Time LLM Training Dashboard")
    print("=" * 60)

    if torch and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', 0) / 1024 / 1024
        print(f"  GPU: {gpu} ({vram:.0f} MB)")
    else:
        print("  GPU: None (CPU only)")

    if not args.no_model:
        model_path = require_existing_model_path(config)
        print(f"  Model: {model_path}")
        print(f"  Quantization: {config['model']['quantization']}")
        print(f"  LoRA rank: {config['model']['lora_rank']}")
        print()
        print("  Loading model...")

        from model_loader import load_model, get_embedding_layer
        model, tokenizer, peft_config = load_model(config)

        print("  Registering hooks...")
        from hooks import HookManager
        hook_manager = HookManager(model)
        print(f"  Hooks: {hook_manager._snapshot.n_layers} transformer blocks")

        print("  Initializing metric computer...")
        from metrics import MetricComputer
        metric_computer = MetricComputer(model, config)
        initial_emb = get_embedding_layer(model)
        if initial_emb is not None:
            metric_computer.set_initial_embedding(initial_emb)
            print("  Embedding snapshot saved for drift tracking")

        print("  Initializing trainer...")
        from trainer import Trainer
        trainer = Trainer(model, tokenizer, hook_manager, metric_computer, config)

        print("  Initializing biological reference model...")
        from bio_model import IzhikevichNetwork
        bio_model = IzhikevichNetwork(config)
        print(f"  Bio model: {bio_model.n} neurons ({bio_model.n_exc} exc + {bio_model.n_inh} inh)")

        print("  Initializing self-modification engine...")
        from self_modify import SelfModificationEngine
        self_modify_engine = SelfModificationEngine(model, tokenizer, trainer, config, hook_manager)
        print(f"  Self-modify: {'enabled' if self_modify_engine.enabled else 'disabled'} "
              f"(every {self_modify_engine.eval_every} steps)")

        # Wire components into server
        from server import set_components
        set_components(trainer, bio_model, self_modify_engine, metric_computer, config)
    else:
        print("  Running in dashboard-only mode (no model loaded)")
        print()

    if torch and torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024 / 1024
        vram_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"  VRAM: {vram_used:.0f} MB allocated, {vram_reserved:.0f} MB reserved")

    print()
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  WebSocket: ws://localhost:{port}/ws")
    print("=" * 60)
    print()

    from server import app
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
