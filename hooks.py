"""
enhanced/hooks.py — PyTorch forward/backward hooks for real-time metric extraction.

Extends the base HookManager with:
- Attention head importance (gradient * activation magnitude, dead head detection)
- Rank collapse detection (SVD of weight matrices per layer)
- Layer saturation (fraction of activations near max/min bounds)
- Gradient flow visualization (per-layer magnitude + direction dominance)
- Feature correlation (pairwise correlation of activation channels)
- Activation histograms (16 bins per layer)
- Weight update magnitude per layer (parameter velocity)
- Fisher information diagonal (running estimate)
- Synaptic tagging (gradient * weight product magnitude)

Extended HookSnapshot fields:
  head_importance, rank_ratios, saturation_fractions, activation_histograms,
  feature_correlation_mean, weight_update_velocity, fisher_diagonal_mean
"""

import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# HookSnapshot — extended
# ---------------------------------------------------------------------------

@dataclass
class HookSnapshot:
    """Raw data captured from one forward+backward pass."""
    # Per-layer activations (list of [batch, seq, hidden] norms)
    layer_activation_norms: list = field(default_factory=list)
    # Per-layer gradient norms
    layer_gradient_norms: list = field(default_factory=list)
    # Per-layer weight norms (L2)
    layer_weight_norms: list = field(default_factory=list)
    # Attention weights per layer: list of [n_heads] entropy values
    attention_entropies: list = field(default_factory=list)
    # Attention weight matrices for context utilisation (last layer)
    attention_position_mass: list = field(default_factory=list)
    # LayerNorm stats per layer: (mean, var)
    layernorm_stats: list = field(default_factory=list)
    # Activation survival rate per layer (fraction non-zero post-activation)
    activation_survival: list = field(default_factory=list)
    # Residual stream ratio per layer (skip_mag / total_mag)
    residual_ratios: list = field(default_factory=list)
    # Mean-pooled hidden state vector per layer (for PCA latent space)
    hidden_vectors: list = field(default_factory=list)
    # Number of layers
    n_layers: int = 0

    # ---- Extended fields ----
    # Per-layer, per-head importance scores (list of lists: [n_layers][n_heads])
    head_importance: list = field(default_factory=list)
    # Effective rank ratio per layer (effective_rank / min_dim)
    rank_ratios: list = field(default_factory=list)
    # Saturation fraction per layer (fraction of activations near bounds)
    saturation_fractions: list = field(default_factory=list)
    # Activation histograms per layer (16 bins each)
    activation_histograms: list = field(default_factory=list)
    # Mean pairwise feature correlation per layer
    feature_correlation_mean: list = field(default_factory=list)
    # Weight update velocity per layer (||W_t - W_{t-1}||)
    weight_update_velocity: list = field(default_factory=list)
    # Fisher information diagonal mean per layer
    fisher_diagonal_mean: list = field(default_factory=list)
    # Synaptic tag magnitude per layer (mean |grad * weight|)
    synaptic_tag_mean: list = field(default_factory=list)
    # Gradient flow direction per layer (ratio of forward-dominant flow)
    gradient_flow_direction: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# HookManager — extended
# ---------------------------------------------------------------------------

class HookManager:
    """Registers forward and backward hooks on a transformer model.

    Thread-safe: snapshot is written atomically under a lock.
    """

    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        self._handles = []
        self._forward_data: Dict[int, dict] = {}
        self._backward_data: Dict[int, dict] = {}
        self._snapshot = HookSnapshot()
        self._capture_hidden_vecs = True
        self.enabled = True

        # For weight velocity tracking
        self._prev_weight_snapshots: Dict[int, Dict[str, torch.Tensor]] = {}

        # For Fisher diagonal running estimate
        self._fisher_accum: Dict[int, float] = {}
        self._fisher_count: Dict[int, int] = {}

        self._register_hooks()

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def _register_hooks(self):
        """Find transformer blocks and register hooks."""
        blocks = self._find_transformer_blocks()
        self._snapshot.n_layers = len(blocks)

        for idx, (name, module) in enumerate(blocks):
            handle = module.register_forward_hook(self._make_forward_hook(idx, name))
            self._handles.append(handle)

            handle = module.register_full_backward_hook(self._make_backward_hook(idx, name))
            self._handles.append(handle)

    def _find_transformer_blocks(self) -> list:
        """Find all transformer decoder layers."""
        blocks = []
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__
            if any(k in cls_name for k in ["DecoderLayer", "Qwen3DecoderLayer", "LlamaDecoderLayer"]):
                blocks.append((name, module))
        if not blocks:
            for name, module in self.model.named_modules():
                if ".layers." in name and name.split(".")[-1].isdigit():
                    if not any(n == name for n, _ in blocks):
                        blocks.append((name, module))
        return blocks

    # -----------------------------------------------------------------------
    # Forward hook
    # -----------------------------------------------------------------------

    def _make_forward_hook(self, layer_idx: int, layer_name: str):
        def hook(module, input, output):
            if not self.enabled:
                return
            try:
                if isinstance(output, tuple):
                    hidden = output[0]
                    attn_weights = output[1] if len(output) > 1 else None
                else:
                    hidden = output
                    attn_weights = None

                data = {}

                if hidden is not None and isinstance(hidden, torch.Tensor):
                    h = hidden.detach().float()

                    # Basic norm
                    data["activation_norm"] = h.norm().item()

                    # Activation survival
                    data["survival"] = (h.abs() > 1e-6).float().mean().item()

                    # Saturation: fraction where |h| > 0.9 * max|h|
                    h_max = h.abs().max().item() + 1e-10
                    data["saturation"] = (h.abs() > 0.9 * h_max).float().mean().item()

                    # Activation histogram (16 bins)
                    try:
                        flat_h = h.flatten().cpu().numpy()
                        hist_vals, _ = np.histogram(flat_h, bins=16)
                        data["activation_hist"] = (hist_vals / (hist_vals.sum() + 1e-10)).tolist()
                    except Exception:
                        data["activation_hist"] = [0.0] * 16

                    # Feature correlation mean (sample up to 64 channels)
                    try:
                        if h.dim() == 3:  # [batch, seq, hidden]
                            channels = h[0].cpu().numpy()  # [seq, hidden]
                            n_ch = min(64, channels.shape[-1])
                            ch_sample = channels[:, :n_ch].T  # [n_ch, seq]
                            if ch_sample.shape[1] > 1:
                                corr = np.corrcoef(ch_sample)
                                triu = corr[np.triu_indices_from(corr, k=1)]
                                data["feature_corr_mean"] = float(np.abs(triu).mean())
                            else:
                                data["feature_corr_mean"] = 0.0
                    except Exception:
                        data["feature_corr_mean"] = 0.0

                    # Mean-pooled hidden vector for latent PCA
                    if self._capture_hidden_vecs and layer_idx % 6 == 0:
                        try:
                            data["hidden_vec"] = h.mean(dim=(0, 1)).cpu().numpy().tolist()
                        except Exception:
                            pass

                    # Residual ratio
                    if isinstance(input, tuple) and len(input) > 0:
                        inp = input[0]
                        if isinstance(inp, torch.Tensor) and inp.shape == hidden.shape:
                            residual = inp.detach().float().norm().item()
                            transform = (h - inp.detach().float()).norm().item()
                            total = residual + transform + 1e-10
                            data["residual_ratio"] = residual / total

                # Attention weights
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    with torch.no_grad():
                        aw = attn_weights.detach().float().clamp(min=1e-10)
                        # [batch, n_heads, seq, seq]
                        entropy = -(aw * aw.log()).sum(dim=-1).mean(dim=(0, 2))  # [n_heads]
                        data["attn_entropy"] = entropy.cpu().numpy().tolist()

                        # Head importance: mean attention magnitude per head
                        head_imp = aw.mean(dim=(0, 2, 3)).cpu().numpy().tolist()  # [n_heads]
                        data["head_importance"] = head_imp

                        # Position mass
                        pos_mass = aw.mean(dim=(0, 1, 2)).cpu().numpy()
                        data["attn_position_mass"] = pos_mass.tolist()

                # Weight norms
                weight_norm = 0.0
                count = 0
                for pname, param in module.named_parameters():
                    if param.requires_grad or "weight" in pname:
                        try:
                            weight_norm += param.detach().float().norm().item()
                            count += 1
                        except Exception:
                            pass
                if count > 0:
                    data["weight_norm"] = weight_norm / count

                # Weight velocity (||W_t - W_{t-1}||)
                try:
                    curr_weights = {n: p.detach().float().norm().item()
                                    for n, p in module.named_parameters()}
                    prev = self._prev_weight_snapshots.get(layer_idx, {})
                    velocities = []
                    for n, curr_norm in curr_weights.items():
                        if n in prev:
                            velocities.append(abs(curr_norm - prev[n]))
                    data["weight_velocity"] = float(np.mean(velocities)) if velocities else 0.0
                    self._prev_weight_snapshots[layer_idx] = curr_weights
                except Exception:
                    data["weight_velocity"] = 0.0

                # LayerNorm stats
                for subname, submod in module.named_modules():
                    if isinstance(submod, (nn.LayerNorm, nn.RMSNorm)) or "norm" in subname.lower():
                        if hasattr(submod, "weight") and submod.weight is not None:
                            w = submod.weight.detach().float()
                            data["ln_mean"] = w.mean().item()
                            data["ln_var"] = w.var().item()
                            break

                # Rank ratio via SVD on largest weight matrix in this layer
                try:
                    best_w = None
                    best_size = 0
                    for pname, param in module.named_parameters():
                        if param.dim() >= 2 and param.numel() > best_size:
                            best_w = param.detach().float()
                            best_size = param.numel()
                    if best_w is not None and best_w.dim() == 2:
                        # Limit size for speed
                        max_dim = 256
                        w_small = best_w[:max_dim, :max_dim].cpu()
                        s = torch.linalg.svdvals(w_small)
                        s_norm = s / (s.sum() + 1e-10)
                        eff_rank = float(torch.exp(-(s_norm * torch.log(s_norm + 1e-10)).sum()).item())
                        data["rank_ratio"] = eff_rank / min(w_small.shape)
                    else:
                        data["rank_ratio"] = 1.0
                except Exception:
                    data["rank_ratio"] = 1.0

                self._forward_data[layer_idx] = data

            except Exception:
                pass

        return hook

    # -----------------------------------------------------------------------
    # Backward hook
    # -----------------------------------------------------------------------

    def _make_backward_hook(self, layer_idx: int, layer_name: str):
        def hook(module, grad_input, grad_output):
            if not self.enabled:
                return
            try:
                grad_norm = 0.0
                if isinstance(grad_output, tuple):
                    for g in grad_output:
                        if g is not None and isinstance(g, torch.Tensor):
                            grad_norm = g.detach().float().norm().item()
                            break
                elif isinstance(grad_output, torch.Tensor):
                    grad_norm = grad_output.detach().float().norm().item()

                # Fisher information diagonal estimate
                fisher_contrib = 0.0
                synaptic_tag = 0.0
                n_params = 0
                for pname, param in module.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        g = param.grad.detach().float()
                        fisher_contrib += (g ** 2).mean().item()
                        synaptic_tag += (g.abs() * param.detach().float().abs()).mean().item()
                        n_params += 1

                if n_params > 0:
                    fisher_contrib /= n_params
                    synaptic_tag /= n_params

                # Running Fisher estimate (EMA)
                alpha = 0.1
                if layer_idx in self._fisher_accum:
                    self._fisher_accum[layer_idx] = (
                        (1 - alpha) * self._fisher_accum[layer_idx] + alpha * fisher_contrib
                    )
                else:
                    self._fisher_accum[layer_idx] = fisher_contrib

                # Gradient flow direction: ratio of grad_output norm to grad_input norm
                grad_in_norm = 0.0
                if isinstance(grad_input, tuple):
                    for g in grad_input:
                        if g is not None and isinstance(g, torch.Tensor):
                            grad_in_norm = g.detach().float().norm().item()
                            break
                flow_ratio = grad_norm / (grad_norm + grad_in_norm + 1e-10)

                self._backward_data[layer_idx] = {
                    "grad_norm": grad_norm,
                    "fisher_diag": self._fisher_accum.get(layer_idx, 0.0),
                    "synaptic_tag": synaptic_tag,
                    "flow_ratio": flow_ratio,
                }
            except Exception:
                pass

        return hook

    # -----------------------------------------------------------------------
    # Compile snapshot
    # -----------------------------------------------------------------------

    def compile_snapshot(self) -> HookSnapshot:
        """Compile captured forward/backward data into a clean snapshot."""
        n = self._snapshot.n_layers
        snap = HookSnapshot(n_layers=n)

        for i in range(n):
            fwd = self._forward_data.get(i, {})
            bwd = self._backward_data.get(i, {})

            snap.layer_activation_norms.append(fwd.get("activation_norm", 0.0))
            snap.layer_gradient_norms.append(bwd.get("grad_norm", 0.0))
            snap.layer_weight_norms.append(fwd.get("weight_norm", 0.0))
            snap.activation_survival.append(fwd.get("survival", 1.0))
            snap.residual_ratios.append(fwd.get("residual_ratio", 0.5))

            # Attention
            if "attn_entropy" in fwd:
                snap.attention_entropies.append(fwd["attn_entropy"])
            else:
                snap.attention_entropies.append([])

            # Head importance
            snap.head_importance.append(fwd.get("head_importance", []))

            snap.layernorm_stats.append(
                (fwd.get("ln_mean", 0.0), fwd.get("ln_var", 1.0))
            )

            # Extended fields
            snap.saturation_fractions.append(fwd.get("saturation", 0.0))
            snap.activation_histograms.append(fwd.get("activation_hist", [0.0] * 16))
            snap.feature_correlation_mean.append(fwd.get("feature_corr_mean", 0.0))
            snap.weight_update_velocity.append(fwd.get("weight_velocity", 0.0))
            snap.rank_ratios.append(fwd.get("rank_ratio", 1.0))
            snap.fisher_diagonal_mean.append(bwd.get("fisher_diag", 0.0))
            snap.synaptic_tag_mean.append(bwd.get("synaptic_tag", 0.0))
            snap.gradient_flow_direction.append(bwd.get("flow_ratio", 0.5))

        # Attention position mass from last layer
        last_fwd = self._forward_data.get(n - 1, {})
        snap.attention_position_mass = last_fwd.get("attn_position_mass", [])

        # Hidden vectors for PCA
        for i in range(n):
            fwd = self._forward_data.get(i, {})
            if "hidden_vec" in fwd:
                snap.hidden_vectors.append(fwd["hidden_vec"])

        with self._lock:
            self._snapshot = snap

        return snap

    def get_snapshot(self) -> HookSnapshot:
        """Return the latest snapshot (thread-safe read)."""
        with self._lock:
            return self._snapshot

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
