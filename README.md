# Neural Simulation — Real-Time LLM Training Dashboard

A real-time 3D visualization and training control system for large language models. Two brain visualizations — a biological Izhikevich spiking network and an actual Qwen3-8B loaded via HuggingFace — side by side in a browser, with 24 diagnostic panels, interactive parameter sliders, simultaneous inference + training, and adaptive self-modification.

## Architecture

| Component | Technology |
|---|---|
| AI Model | Qwen3-8B, 4-bit NF4 quantization, LoRA rank-16 adapter |
| Bio Model | Izhikevich spiking network (100 neurons, STDP, homeostasis) |
| 3D Visualization | Three.js instanced spheres with Unreal Bloom |
| Diagnostic Panels | Chart.js sparklines (24 panels, bio=blue vs AI=orange) |
| Backend | FastAPI + WebSocket (runs in WSL for CUDA) |
| Frontend | HTML/CSS/JS served as static files |
| Self-Modification | LLM analyzes its own diagnostics and adjusts training params |

## Requirements

- WSL2 with CUDA support
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 5070 Ti)
- Python 3.12+ in WSL
- Qwen3-8B model at `/home/l3ung/Qwen3/` (or Llama3.1-8B at `/home/l3ung/Llama3/`)

## Installation (WSL)

```bash
cd /mnt/c/Users/l3ung/Desktop/ReverseEngineering/toolbox/neural-sim
pip install -r requirements.txt
```

## Usage

```bash
# Full launch (loads model + starts dashboard)
python launch.py

# Custom model path
python launch.py --model /path/to/model

# Dashboard-only mode (no GPU needed)
python launch.py --no-model

# Custom port
python launch.py --port 9000
```

Then open `http://localhost:8765` in a browser on Windows.

## Dashboard Layout

```
┌──────────────────────────────────────────────────────┐
│  NEURAL SIMULATION — Status bar, FPS, GPU info       │
├───────────────────────┬──────────────────────────────┤
│  🧠 BIOLOGICAL BRAIN  │  🤖 ARTIFICIAL MIND          │
│  3D — orbit controls  │  3D — orbit controls         │
├───────────────────────┴──────────────────────────────┤
│  24 Diagnostic Panels (6×4 grid, sparklines)         │
├──────────────────────────────────────────────────────┤
│  [Sliders: LR, Dropout, WD, Clip, Temp, EWC]        │
│  [Training input] [Chat] [Self-Modification Log]     │
└──────────────────────────────────────────────────────┘
```

## 24 Diagnostic Panels

| # | Panel | Bio Source | AI Source |
|---|---|---|---|
| 1 | Connectivity / Weight | Mean excitatory weight | Mean layer weight norm |
| 2 | Learning Curve | Firing rate organization | Training loss |
| 3 | Error / Prediction Delta | Firing rate error | Loss delta |
| 4 | Sparsity / Mutual Info | Silent neuron fraction | Mean activation norm |
| 5 | Gradient / Dendritic | Per-group dendritic signal | Per-layer gradient norms |
| 6 | Latent / Place Cells | Place cell 2D clusters | PCA of hidden states |
| 7 | Activation Survival | Active neuron fraction | Post-activation non-zero fraction |
| 8 | Attention Entropy | Selective attention focus | Mean head entropy |
| 9 | Weight Norms | Homeostatic group norms | Per-layer weight norms |
| 10 | Loss Landscape / Stability | Attractor variance | Loss variance window |
| 11 | Normalization Stats | Mean firing rate | LayerNorm weight means |
| 12 | Confidence / Certainty | Pattern certainty | 1 - prediction entropy |
| 13 | Momentum / Habit | Habit strength | AdamW exp_avg momentum |
| 14 | Learning Rate / Plasticity | Critical period decay | Scheduler LR |
| 15 | Sparsity / Synapse Density | Non-zero synapse fraction | LoRA weight sparsity |
| 16 | Residual / Bypass | Direct path ratio | Residual stream ratio |
| 17 | Embedding / Cortical Drift | Cortical stability | Embedding cosine drift |
| 18 | Context / Working Memory | Primacy-recency profile | Attention position mass |
| 19 | Grad Clip / Pain Signal | Extreme firing events | Gradient clip rate |
| 20 | Forgetting / Retention | Memory consolidation | Old-task loss increase |
| 21 | Effective Rank / Diversity | Population entropy | SVD-based effective rank |
| 22 | Gradient SNR / Noise | Stochastic resonance | Mean/std gradient ratio |
| 23 | CKA / Column Diff | Adjacent group differentiation | Adjacent layer norm ratio |
| 24 | Expert / Specialisation | Area activity distribution | Head entropy distribution |

## Self-Modification

Every 25 training steps, the LLM receives a formatted summary of all diagnostics. It can call a `self_modify` tool to adjust one parameter (learning rate, dropout, temperature, gradient clip, EWC lambda, weight decay). Modifications are validated against configurable bounds, applied immediately, logged, and broadcast to the dashboard in real time.

## Files

```
launch.py           — Entry point (uvicorn + component init)
config.yaml         — Model paths, params, bounds
model_loader.py     — NF4 quantization + LoRA setup
hooks.py            — Forward/backward hooks on transformer blocks
metrics.py          — 24-metric computer (AI + Bio)
trainer.py          — LoRA training loop with async inference
bio_model.py        — Izhikevich spiking network
self_modify.py      — Adaptive self-modification engine
server.py           — FastAPI + WebSocket hub
requirements.txt    — Python dependencies
static/
  index.html        — Dashboard shell
  css/dashboard.css — Dark theme
  js/brain3d.js     — Three.js 3D brain visualization
  js/panels.js      — 24 Chart.js diagnostic panels
  js/controls.js    — Sliders, buttons, chat
  js/websocket.js   — Auto-reconnect WebSocket client
  js/app.js         — Main controller
```

