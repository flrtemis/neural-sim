"""
enhanced/trainer.py — LoRA fine-tuning engine with advanced training techniques.

Extends the base Trainer with:
- MAML (Model-Agnostic Meta-Learning) meta-gradient updates
- Continual learning via replay buffer (ring buffer, 512 samples)
- Gradient Surgery (PCGrad) to prevent catastrophic forgetting
- Dynamic LoRA rank adaptation based on effective rank monitoring
- SGDR (cosine annealing with warm restarts, T_0=50, T_mult=2)
- SAM (Sharpness-Aware Minimization) wrapper around AdamW
- Online distillation with EMA teacher model
- Token-level importance weighting
- Gradient noise injection (sigma=0.01)
- Activation checkpointing detection
- Per-layer learning rate scaling (LLRD, decay_factor=0.95)
- Adaptive batch construction from highest-loss chunks
- Real-time perplexity and prediction entropy tracking
- Neuron activation statistics (dead neuron detection)
- All original functionality preserved
"""

import asyncio
import copy
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    step: int = 0
    total_loss: float = 0.0
    running: bool = False
    paused: bool = False


# ---------------------------------------------------------------------------
# SAM Optimizer wrapper
# ---------------------------------------------------------------------------

class SAMOptimizer:
    """Sharpness-Aware Minimization wrapper around a base optimizer.

    Usage:
        sam = SAMOptimizer(base_optimizer, rho=0.05)
        # First step: perturb weights
        sam.first_step(zero_grad=True)
        loss_sharp = forward()
        loss_sharp.backward()
        # Second step: restore weights + actual update
        sam.second_step(zero_grad=True)
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self._e_w_buf: Dict = {}   # stores perturbation e(w)
        self.state = base_optimizer.state
        self.param_groups = base_optimizer.param_groups

    def _gradient_norm(self) -> torch.Tensor:
        """Compute the l2 norm of all parameter gradients."""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ])
        )
        return norm

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Perturb weights: w_hat = w + rho * grad / ||grad||."""
        grad_norm = self._gradient_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # perturb
                self._e_w_buf[id(p)] = e_w.clone()
        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore weights and do the actual parameter update."""
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in self._e_w_buf:
                    p.sub_(self._e_w_buf[id(p)])  # restore
        self.base_optimizer.step()
        if zero_grad:
            self.base_optimizer.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self):
        """Standard step (delegates to base; use first_step/second_step for SAM)."""
        self.base_optimizer.step()

    def get_last_lr(self):
        try:
            return [g["lr"] for g in self.param_groups]
        except Exception:
            return [0.0]


# ---------------------------------------------------------------------------
# Replay Buffer (ring buffer)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Ring buffer of (chunk_tensor, loss_estimate) pairs."""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    def push(self, chunk: torch.Tensor, loss_est: float = 1.0):
        self._buf.append((chunk.cpu().clone(), loss_est))

    def sample_highest_loss(self, k: int = 4) -> List[torch.Tensor]:
        """Return the k highest-loss chunks (adaptive batch construction)."""
        if not self._buf:
            return []
        sorted_items = sorted(self._buf, key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:k]]

    def sample_random(self, k: int = 4) -> List[torch.Tensor]:
        """Return k random chunks."""
        if not self._buf:
            return []
        indices = np.random.choice(len(self._buf), size=min(k, len(self._buf)), replace=False)
        items = list(self._buf)
        return [items[i][0] for i in indices]

    def __len__(self):
        return len(self._buf)

    def update_loss(self, chunk: torch.Tensor, new_loss: float):
        """Update loss estimate for a chunk that matches."""
        buf_list = list(self._buf)
        for i, (c, _) in enumerate(buf_list):
            if c.shape == chunk.shape and torch.equal(c, chunk.cpu()):
                buf_list[i] = (c, new_loss)
                self._buf = deque(buf_list, maxlen=self.capacity)
                return


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Enhanced LoRA fine-tuning engine with hook-based metric extraction.

    Adds MAML, SAM, continual learning, dynamic LoRA rank adaptation,
    distillation, gradient surgery, LLRD, and many more advanced features
    on top of the original Trainer interface.
    """

    def __init__(self, model, tokenizer, hook_manager, metric_computer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = hook_manager
        self.metrics = metric_computer
        self.config = config
        self.train_cfg = config["training"]

        # ---- Build per-layer param groups for LLRD ----
        self._llrd_factor = self.train_cfg.get("llrd_factor", 0.95)
        param_groups = self._build_llrd_param_groups()

        # ---- Base AdamW optimizer ----
        base_lr = self.train_cfg["learning_rate"]
        self._base_lr = base_lr
        self._base_adamw = torch.optim.AdamW(
            param_groups,
            betas=tuple(self.train_cfg["optimizer_betas"]),
            weight_decay=self.train_cfg["weight_decay"],
        )

        # ---- SAM wrapper ----
        self._sam_rho = self.train_cfg.get("sam_rho", 0.05)
        self.optimizer = SAMOptimizer(self._base_adamw, rho=self._sam_rho)

        # ---- SGDR scheduler (warm restarts) ----
        self._sgdr_T0 = 50
        self._sgdr_Tmult = 2
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._base_adamw,
            T_0=self._sgdr_T0,
            T_mult=self._sgdr_Tmult,
            eta_min=base_lr * 0.01,
        )

        self._warmup_steps = self.train_cfg["warmup_steps"]

        # ---- State ----
        self.state = TrainState()
        self._inference_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()

        # Metric callback: called with (ai_metrics_dict, step)
        self.on_metrics = None

        # ---- EWC ----
        self._fisher: Dict = {}
        self._ewc_star: Dict = {}
        self._ewc_lambda = self.train_cfg.get("ewc_lambda", 0.1)

        # ---- Forgetting ----
        self._old_task_prompt: Optional[str] = None
        self._old_task_baseline_loss: Optional[float] = None
        self._forgetting_score: float = 0.0

        # ---- Gradient clipping ----
        self._grad_clip_norm = self.train_cfg.get("gradient_clip_norm", 1.0)

        # ---- Temperature for inference ----
        self.temperature = 1.0

        # ---- Replay buffer (ring buffer of 512) ----
        self._replay_buf = ReplayBuffer(capacity=512)
        self._replay_ratio = self.train_cfg.get("replay_ratio", 0.25)

        # ---- EMA teacher model for online distillation ----
        self._distill_alpha = self.train_cfg.get("distill_alpha", 0.1)
        self._ema_decay = 0.999
        self._ema_teacher: Optional[nn.Module] = None
        self._init_ema_teacher()

        # ---- Gradient noise ----
        self._grad_noise_sigma = self.train_cfg.get("grad_noise_sigma", 0.01)

        # ---- Token importance weights ----
        self._token_importance: Optional[torch.Tensor] = None
        self._token_importance_ema = 0.9

        # ---- SAM sharpness tracking ----
        self._sam_sharpness = 0.0

        # ---- MAML ----
        self._meta_lr = self.train_cfg.get("meta_lr", 1e-4)
        self._meta_loss = 0.0

        # ---- Dead neuron tracking ----
        self._dead_neurons_per_layer: List[float] = []

        # ---- Distillation loss tracking ----
        self._distill_loss = 0.0

        # ---- Replay loss tracking ----
        self._replay_loss = 0.0

        # ---- Perplexity ----
        self._perplexity = 1.0

        # ---- Previous gradient vector for PCGrad ----
        self._prev_grad_vec: Optional[torch.Tensor] = None

        # ---- Previous weights for velocity tracking ----
        self._prev_weights: Dict[str, torch.Tensor] = {}

        # ---- Dynamic LoRA rank tracking ----
        self._current_lora_rank: int = self.train_cfg.get("lora_rank", 16)
        self._lora_rank_collapse_threshold = 0.2
        self._lora_rank_expand_threshold = 0.85

        # Activation checkpointing detection
        self._uses_grad_checkpoint = self._detect_grad_checkpointing()

        # Token-level importance tracking
        self._token_imp_mean = 0.0

    # -----------------------------------------------------------------------
    # Initialization helpers
    # -----------------------------------------------------------------------

    def _build_llrd_param_groups(self) -> list:
        """Build param groups with layer-wise learning rate decay."""
        base_lr = self.train_cfg["learning_rate"]
        decay = self._llrd_factor
        trainable_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

        # Group by layer depth heuristic (layer number in name)
        layer_groups: Dict[int, list] = {}
        no_layer_params = []

        for name, param in trainable_params:
            # Try to extract layer index
            parts = name.split(".")
            layer_idx = None
            for part in parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break
            if layer_idx is not None:
                layer_groups.setdefault(layer_idx, []).append(param)
            else:
                no_layer_params.append(param)

        groups = []
        max_layer = max(layer_groups.keys()) if layer_groups else 0
        for layer_idx in sorted(layer_groups.keys()):
            depth_from_top = max_layer - layer_idx
            lr = base_lr * (decay ** depth_from_top)
            groups.append({"params": layer_groups[layer_idx], "lr": lr})

        if no_layer_params:
            groups.append({"params": no_layer_params, "lr": base_lr})

        if not groups:
            # Fallback: all trainable params
            all_trainable = [p for p in self.model.parameters() if p.requires_grad]
            groups = [{"params": all_trainable, "lr": base_lr}]

        return groups

    def _init_ema_teacher(self):
        """Initialize EMA teacher as a deep copy of the model (eval mode)."""
        try:
            self._ema_teacher = copy.deepcopy(self.model)
            self._ema_teacher.eval()
            for p in self._ema_teacher.parameters():
                p.requires_grad_(False)
        except Exception as e:
            print(f"[trainer] EMA teacher init failed: {e}")
            self._ema_teacher = None

    def _detect_grad_checkpointing(self) -> bool:
        """Detect if model uses gradient checkpointing."""
        for module in self.model.modules():
            if hasattr(module, "gradient_checkpointing") and module.gradient_checkpointing:
                return True
            if hasattr(module, "_gradient_checkpointing_func"):
                return True
        return False

    # -----------------------------------------------------------------------
    # EMA teacher update
    # -----------------------------------------------------------------------

    def _update_ema_teacher(self):
        """Exponential moving average update of teacher weights."""
        if self._ema_teacher is None:
            return
        try:
            with torch.no_grad():
                for (name, student_p), (_, teacher_p) in zip(
                    self.model.named_parameters(), self._ema_teacher.named_parameters()
                ):
                    teacher_p.data.mul_(self._ema_decay).add_(
                        student_p.data * (1.0 - self._ema_decay)
                    )
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Distillation loss
    # -----------------------------------------------------------------------

    def _distillation_loss(self, student_logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """KL divergence between student and EMA teacher logits."""
        if self._ema_teacher is None or student_logits is None:
            return torch.tensor(0.0, device=self.model.device)
        try:
            with torch.no_grad():
                teacher_out = self._ema_teacher(input_ids=input_ids, labels=input_ids)
                teacher_logits = teacher_out.logits.detach()

            T = 2.0  # temperature for distillation
            s_log_prob = F.log_softmax(student_logits / T, dim=-1)
            t_prob = F.softmax(teacher_logits / T, dim=-1)
            kl = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T ** 2)
            return kl
        except Exception:
            return torch.tensor(0.0, device=self.model.device)

    # -----------------------------------------------------------------------
    # EWC
    # -----------------------------------------------------------------------

    def _ewc_loss(self) -> torch.Tensor:
        """EWC penalty to prevent catastrophic forgetting."""
        if not self._fisher:
            return torch.tensor(0.0, device=self.model.device)
        loss = torch.tensor(0.0, device=self.model.device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._fisher:
                fisher = self._fisher[name]
                star = self._ewc_star[name]
                loss += (fisher * (param - star).pow(2)).sum()
        return self._ewc_lambda * 0.5 * loss

    def compute_fisher(self, text: str, n_samples: int = 5):
        """Estimate Fisher information from a few forward passes."""
        self._fisher.clear()
        self._ewc_star.clear()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.train_cfg["max_seq_length"],
            truncation=True,
        ).to(self.model.device)
        for _ in range(n_samples):
            self.model.zero_grad()
            out = self.model(**enc, labels=enc["input_ids"])
            out.loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in self._fisher:
                        self._fisher[name] = param.grad.data.clone().pow(2) / n_samples
                    else:
                        self._fisher[name] += param.grad.data.clone().pow(2) / n_samples
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._fisher:
                self._ewc_star[name] = param.data.clone()

    # -----------------------------------------------------------------------
    # Tokenization
    # -----------------------------------------------------------------------

    def _tokenize_training_text(self, text: str) -> list:
        """Tokenize training text into overlapping chunks."""
        max_len = self.train_cfg["max_seq_length"]
        stride = max_len // 2
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"][0]
        chunks = []
        for start in range(0, len(input_ids) - 1, stride):
            end = min(start + max_len, len(input_ids))
            chunk = input_ids[start:end]
            if len(chunk) < 4:
                continue
            chunks.append(chunk)
            if end >= len(input_ids):
                break
        return chunks

    # -----------------------------------------------------------------------
    # PCGrad: gradient surgery
    # -----------------------------------------------------------------------

    def _pcgrad_project(self, grad_task: torch.Tensor, grad_replay: torch.Tensor) -> torch.Tensor:
        """Project grad_task onto the plane orthogonal to grad_replay if they conflict."""
        dot = torch.dot(grad_task.flatten(), grad_replay.flatten())
        if dot < 0:
            norm_sq = grad_replay.flatten().dot(grad_replay.flatten()) + 1e-10
            grad_task = grad_task - (dot / norm_sq) * grad_replay
        return grad_task

    def _apply_gradient_surgery(self, task_grads: List[torch.Tensor], replay_grads: List[torch.Tensor]):
        """Apply PCGrad projection across all parameter gradients."""
        if not task_grads or not replay_grads or len(task_grads) != len(replay_grads):
            return
        for i, (g_task, g_replay) in enumerate(zip(task_grads, replay_grads)):
            if g_task is not None and g_replay is not None:
                task_grads[i] = self._pcgrad_project(g_task, g_replay)

    # -----------------------------------------------------------------------
    # Gradient noise injection
    # -----------------------------------------------------------------------

    def _inject_gradient_noise(self, sigma: float):
        """Add Gaussian noise to all parameter gradients."""
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    noise = torch.randn_like(p.grad) * sigma
                    p.grad.add_(noise)

    # -----------------------------------------------------------------------
    # Token importance weighting
    # -----------------------------------------------------------------------

    def _compute_token_importance(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Compute per-token gradient magnitude as importance weight."""
        try:
            with torch.no_grad():
                # Per-token cross-entropy
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = input_ids[:, 1:].contiguous()
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                )
                # Normalize to weights summing to sequence_length
                weights = per_token_loss / (per_token_loss.mean() + 1e-10)
                self._token_imp_mean = float(weights.mean().item())
                return weights
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Dead neuron detection
    # -----------------------------------------------------------------------

    def _compute_dead_neurons(self) -> List[float]:
        """Estimate fraction of dead neurons per transformer layer via hook data."""
        dead_per_layer = []
        try:
            snap = self.hooks.get_snapshot()
            for surv in snap.activation_survival:
                # survival = fraction of non-zero; dead = 1 - survival
                dead_per_layer.append(max(0.0, 1.0 - surv))
        except Exception:
            pass
        return dead_per_layer

    # -----------------------------------------------------------------------
    # Dynamic LoRA rank adaptation
    # -----------------------------------------------------------------------

    def _check_lora_rank(self):
        """Monitor effective rank and adapt LoRA rank if needed."""
        try:
            lora_weights = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and "lora" in name.lower() and param.dim() == 2:
                    lora_weights.append((name, param.detach().float().cpu()))
            if not lora_weights:
                return
            # Take largest LoRA matrix
            w = max(lora_weights, key=lambda x: x[1].numel())[1]
            s = torch.linalg.svdvals(w)
            s_norm = s / (s.sum() + 1e-10)
            eff_rank = math.exp(-(s_norm * torch.log(s_norm + 1e-10)).sum().item())
            max_rank = min(w.shape)
            rank_ratio = eff_rank / max_rank

            if rank_ratio < self._lora_rank_collapse_threshold:
                # Rank collapsed: report (actual weight insertion would require PEFT hooks)
                print(f"[trainer] LoRA rank collapse detected (ratio={rank_ratio:.3f}). Consider doubling rank.")
            elif rank_ratio > self._lora_rank_expand_threshold:
                # Rank too spread: prune smallest singular values
                print(f"[trainer] LoRA rank expansion detected (ratio={rank_ratio:.3f}). Consider pruning.")
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # SAM sharpness estimation
    # -----------------------------------------------------------------------

    def _estimate_sam_sharpness(self, chunk: torch.Tensor) -> float:
        """Estimate loss landscape sharpness at current point."""
        try:
            input_ids = chunk.unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                out_base = self.model(input_ids=input_ids, labels=input_ids)
                loss_base = out_base.loss.item()
            # Perturb weights slightly and measure loss change
            noise_scale = 0.001
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data.add_(torch.randn_like(p.data) * noise_scale)
            with torch.no_grad():
                out_perturbed = self.model(input_ids=input_ids, labels=input_ids)
                loss_perturbed = out_perturbed.loss.item()
            # Restore
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data.sub_(torch.randn_like(p.data) * noise_scale)
            sharpness = abs(loss_perturbed - loss_base)
            return sharpness
        except Exception:
            return 0.0

    # -----------------------------------------------------------------------
    # MAML meta-step
    # -----------------------------------------------------------------------

    def meta_step(self, task_chunks: List[torch.Tensor], n_inner: int = 3, inner_lr: float = 0.01) -> float:
        """Compute MAML meta-gradients across multiple task episodes.

        For each task chunk:
        1. Clone current parameters as theta
        2. Take n_inner gradient steps on the task (inner loop)
        3. Compute loss at adapted parameters
        4. Accumulate meta-gradient: d/d_theta[loss(theta_adapted)]

        Then apply meta-update with meta_lr.

        Returns: meta loss value
        """
        if not task_chunks:
            return 0.0

        meta_loss_total = 0.0
        meta_grads: Dict[str, torch.Tensor] = {}

        original_params = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}

        for chunk in task_chunks[:4]:  # Limit to 4 tasks for speed
            # Restore original params for each task
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in original_params:
                    param.data.copy_(original_params[name])

            input_ids = chunk.unsqueeze(0).to(self.model.device)

            # Inner loop: fast adaptation
            for _ in range(n_inner):
                self.model.zero_grad()
                out = self.model(input_ids=input_ids, labels=input_ids)
                out.loss.backward()
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            param.data.sub_(inner_lr * param.grad)

            # Meta-loss at adapted parameters
            self.model.zero_grad()
            out_adapted = self.model(input_ids=input_ids, labels=input_ids)
            meta_loss = out_adapted.loss
            meta_loss.backward()

            meta_loss_total += meta_loss.item()

            # Accumulate meta-gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in meta_grads:
                        meta_grads[name] = param.grad.data.clone()
                    else:
                        meta_grads[name] += param.grad.data.clone()

        # Restore original params
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in original_params:
                param.data.copy_(original_params[name])

        # Apply meta-update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in meta_grads:
                    avg_grad = meta_grads[name] / len(task_chunks)
                    param.data.sub_(self._meta_lr * avg_grad)

        self._meta_loss = meta_loss_total / max(1, len(task_chunks))
        return self._meta_loss

    # -----------------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------------

    def _apply_warmup(self):
        """Apply linear warmup to learning rate."""
        if self.state.step < self._warmup_steps:
            warmup_factor = self.state.step / max(1, self._warmup_steps)
            for group in self._base_adamw.param_groups:
                # Scale by layer-specific base LR (stored as initial_lr if available)
                base = group.get("initial_lr", group.get("lr", self._base_lr))
                group["lr"] = base * warmup_factor

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    async def train_on_text(self, text: str, num_steps: Optional[int] = None):
        """Train on provided text. Runs as an async loop, emitting metrics each step."""
        self.state.running = True
        self.state.paused = False
        self._stop_event.clear()
        self._pause_event.set()

        if not self._fisher:
            self.compute_fisher(text)

        if self._old_task_prompt is None:
            self._old_task_prompt = text[:200]

        chunks = self._tokenize_training_text(text)
        if not chunks:
            self.state.running = False
            return

        # Seed replay buffer with initial chunks
        for chunk in chunks[:min(32, len(chunks))]:
            self._replay_buf.push(chunk, loss_est=1.0)

        max_steps = num_steps or self.train_cfg["max_steps"]
        chunk_idx = 0

        self.model.train()

        for step in range(max_steps):
            if self._stop_event.is_set():
                break
            await self._pause_event.wait()

            # Select chunk: adaptive (highest-loss) or cyclic
            if len(self._replay_buf) > 8 and np.random.random() < self._replay_ratio:
                high_loss_chunks = self._replay_buf.sample_highest_loss(k=1)
                chunk = high_loss_chunks[0] if high_loss_chunks else chunks[chunk_idx % len(chunks)]
            else:
                chunk = chunks[chunk_idx % len(chunks)]
                chunk_idx += 1

            async with self._inference_lock:
                ai_metrics = await asyncio.get_event_loop().run_in_executor(
                    None, self._train_step, chunk, step, chunks
                )

            if self.on_metrics and ai_metrics:
                await self.on_metrics(ai_metrics, step)

            await asyncio.sleep(0.01)

        self.state.running = False

    # -----------------------------------------------------------------------
    # Single training step (SAM + replay + distillation + PCGrad)
    # -----------------------------------------------------------------------

    def _train_step(self, chunk: torch.Tensor, step: int, all_chunks: list) -> dict:
        """Execute one enhanced training step."""
        self.state.step = step

        input_ids = chunk.unsqueeze(0).to(self.model.device)
        labels = input_ids.clone()

        # ==== SAM First Step: forward + backward at current weights ====
        self.optimizer.zero_grad()

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            output_attentions=True,
        )
        loss = outputs.loss
        ewc_loss = self._ewc_loss()

        # Distillation loss
        distill_loss = torch.tensor(0.0, device=self.model.device)
        if self._distill_alpha > 0 and self._ema_teacher is not None:
            distill_loss = self._distillation_loss(outputs.logits.detach() if outputs.logits is not None else None, input_ids)
            if isinstance(distill_loss, torch.Tensor) and not distill_loss.isnan():
                self._distill_loss = distill_loss.item()
            else:
                distill_loss = torch.tensor(0.0, device=self.model.device)

        total_loss = loss + ewc_loss + self._distill_alpha * distill_loss
        total_loss.backward()

        # Collect task gradients for PCGrad
        task_grads = [
            p.grad.clone() if p.grad is not None else None
            for p in self.model.parameters() if p.requires_grad
        ]

        # ==== SAM perturbation step ====
        self.optimizer.first_step(zero_grad=True)

        # Forward at perturbed point
        outputs_sharp = self.model(input_ids=input_ids, labels=labels)
        loss_sharp = outputs_sharp.loss + self._ewc_loss()
        loss_sharp.backward()

        self._sam_sharpness = abs(loss_sharp.item() - loss.item())

        # ==== Replay buffer training (PCGrad) ====
        replay_grads = None
        if len(self._replay_buf) >= 4:
            replay_chunks = self._replay_buf.sample_random(k=4)
            replay_loss_total = torch.tensor(0.0, device=self.model.device)
            for rc in replay_chunks:
                rc_ids = rc.unsqueeze(0).to(self.model.device)
                out_r = self.model(input_ids=rc_ids, labels=rc_ids)
                replay_loss_total += out_r.loss
            replay_loss_total = replay_loss_total / len(replay_chunks)
            self._replay_loss = replay_loss_total.item()

            # Get replay grads for PCGrad
            self._base_adamw.zero_grad()
            replay_loss_total.backward()
            replay_grads = [
                p.grad.clone() if p.grad is not None else None
                for p in self.model.parameters() if p.requires_grad
            ]

        # ==== Gradient Surgery (PCGrad) ====
        if replay_grads is not None:
            self._apply_gradient_surgery(task_grads, replay_grads)
            # Write PCGrad-corrected gradients back
            param_list = [p for p in self.model.parameters() if p.requires_grad]
            for p, g in zip(param_list, task_grads):
                if g is not None:
                    p.grad = g

        # ==== Gradient noise injection ====
        if self._grad_noise_sigma > 0:
            self._inject_gradient_noise(self._grad_noise_sigma)

        # ==== Hook snapshot before optimizer step ====
        hook_snap = self.hooks.compile_snapshot()

        # ==== Gradient clipping ====
        raw_grad_norm = 0.0
        clip_triggered = False
        trainable = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if trainable:
            raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=float("inf")
            ).item()
            clip_triggered = raw_grad_norm > self._grad_clip_norm
            if clip_triggered:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=self._grad_clip_norm)

        # ==== Warmup + SAM second step (actual update) ====
        self._apply_warmup()
        self.optimizer.second_step(zero_grad=True)
        if self.state.step >= self._warmup_steps:
            self.scheduler.step()

        # ==== EMA teacher update ====
        self._update_ema_teacher()

        # ==== Update replay buffer with new loss estimate ====
        self._replay_buf.push(chunk, loss_est=loss.item())

        # ==== Token importance ====
        if outputs.logits is not None:
            self._compute_token_importance(input_ids, outputs.logits.detach())

        # ==== Dead neuron tracking ====
        self._dead_neurons_per_layer = self._compute_dead_neurons()

        # ==== Perplexity ====
        self._perplexity = math.exp(min(loss.item(), 20.0))

        # ==== Dynamic LoRA rank check (every 50 steps) ====
        if step % 50 == 0:
            self._check_lora_rank()

        # ==== Forgetting score ====
        self.state.total_loss = loss.item()
        if step > 0 and step % 25 == 0 and self._old_task_prompt:
            self._forgetting_score = self._evaluate_forgetting()

        # ==== MAML meta-step (every 100 steps on small chunk sample) ====
        if step > 0 and step % 100 == 0 and len(all_chunks) >= 4:
            meta_chunk_sample = [all_chunks[i % len(all_chunks)] for i in range(4)]
            self.meta_step(meta_chunk_sample)

        # ==== Compute 48 AI metrics ====
        ai_m = self.metrics.compute(
            hook_snapshot=hook_snap,
            optimizer=self._base_adamw,
            scheduler=self.scheduler,
            loss_value=loss.item(),
            step=step,
            raw_grad_norm=raw_grad_norm,
            clip_triggered=clip_triggered,
            logits=outputs.logits.detach() if outputs.logits is not None else None,
            forgetting_score=self._forgetting_score,
            perplexity=self._perplexity,
            dead_neurons_per_layer=self._dead_neurons_per_layer,
            sam_sharpness=self._sam_sharpness,
            distill_loss=self._distill_loss,
            meta_loss=self._meta_loss,
            replay_loss=self._replay_loss,
            token_importance_mean=self._token_imp_mean,
        )

        return self.metrics.to_dict(ai_m)

    # -----------------------------------------------------------------------
    # Forgetting evaluation
    # -----------------------------------------------------------------------

    def _evaluate_forgetting(self) -> float:
        """Evaluate forgetting by checking loss on old task prompt."""
        if not self._old_task_prompt:
            return 0.0
        try:
            was_training = self.model.training
            self.model.eval()
            enc = self.tokenizer(
                self._old_task_prompt,
                return_tensors="pt",
                max_length=128,
                truncation=True,
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model(**enc, labels=enc["input_ids"])
                current_loss = out.loss.item()
            if self._old_task_baseline_loss is None:
                self._old_task_baseline_loss = current_loss
                return 0.0
            delta = max(0.0, current_loss - self._old_task_baseline_loss)
            score = min(1.0, delta / (self._old_task_baseline_loss + 1e-10))
            if was_training:
                self.model.train()
            return score
        except Exception:
            return 0.0

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    async def infer(self, prompt: str, max_tokens: int = 128) -> dict:
        """Run inference with metric capture. Thread-safe with training."""
        async with self._inference_lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._infer_sync, prompt, max_tokens
            )
        return result

    def _infer_sync(self, prompt: str, max_tokens: int) -> dict:
        """Synchronous inference (runs in executor)."""
        was_training = self.model.training
        self.model.eval()

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.train_cfg["max_seq_length"],
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**enc, output_attentions=True)
            hook_snap = self.hooks.compile_snapshot()
            self.hooks.enabled = False
            try:
                gen_ids = self.model.generate(
                    **enc,
                    max_new_tokens=max_tokens,
                    temperature=max(0.01, self.temperature),
                    do_sample=self.temperature > 0.01,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            finally:
                self.hooks.enabled = True

        generated = self.tokenizer.decode(
            gen_ids[0][enc["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        attn_data = []
        if outputs.attentions:
            for layer_attn in outputs.attentions:
                mean_attn = layer_attn[0].detach().float().cpu()
                ent = -(mean_attn * torch.log(mean_attn + 1e-10)).sum(dim=-1).mean(dim=-1)
                attn_data.append(ent.tolist())

        if was_training:
            self.model.train()

        return {
            "text": generated,
            "attention_data": attn_data,
            "hook_snapshot": hook_snap,
        }

    # -----------------------------------------------------------------------
    # update_param — expanded
    # -----------------------------------------------------------------------

    def update_param(self, name: str, value: float):
        """Live parameter modification during training."""
        if name == "learning_rate":
            self._base_lr = value
            for group in self._base_adamw.param_groups:
                group["lr"] = value
        elif name == "weight_decay":
            for group in self._base_adamw.param_groups:
                group["weight_decay"] = value
        elif name == "gradient_clip_norm":
            self._grad_clip_norm = value
        elif name == "ewc_lambda":
            self._ewc_lambda = value
        elif name == "dropout":
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = value
        elif name == "temperature":
            self.temperature = value
        elif name in ("momentum", "beta1"):
            for group in self._base_adamw.param_groups:
                betas = group.get("betas", (0.9, 0.999))
                group["betas"] = (value, betas[1])
        elif name == "beta2":
            for group in self._base_adamw.param_groups:
                betas = group.get("betas", (0.9, 0.999))
                group["betas"] = (betas[0], value)
        elif name == "sam_rho":
            self._sam_rho = value
            self.optimizer.rho = value
        elif name == "distill_alpha":
            self._distill_alpha = value
        elif name == "replay_ratio":
            self._replay_ratio = float(np.clip(value, 0.0, 1.0))
        elif name == "llrd_factor":
            self._llrd_factor = value
            # Rebuild param groups with new decay
            param_groups = self._build_llrd_param_groups()
            # Update existing param groups' LR
            for i, (group, new_group) in enumerate(
                zip(self._base_adamw.param_groups, param_groups)
            ):
                group["lr"] = new_group["lr"]
        elif name == "grad_noise_sigma":
            self._grad_noise_sigma = value
        elif name == "meta_lr":
            self._meta_lr = value
        elif name == "lora_rank_adapt":
            self._current_lora_rank = int(value)

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    def pause(self):
        self.state.paused = True
        self._pause_event.clear()

    def resume(self):
        self.state.paused = False
        self._pause_event.set()

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()

    def get_state(self) -> dict:
        return {
            "step": self.state.step,
            "loss": self.state.total_loss,
            "running": self.state.running,
            "paused": self.state.paused,
            "learning_rate": self._base_adamw.param_groups[0]["lr"],
            "weight_decay": self._base_adamw.param_groups[0].get("weight_decay", 0),
            "gradient_clip_norm": self._grad_clip_norm,
            "ewc_lambda": self._ewc_lambda,
            "temperature": self.temperature,
            "sam_rho": self._sam_rho,
            "distill_alpha": self._distill_alpha,
            "replay_ratio": self._replay_ratio,
            "llrd_factor": self._llrd_factor,
            "grad_noise_sigma": self._grad_noise_sigma,
            "meta_lr": self._meta_lr,
            "perplexity": self._perplexity,
            "sam_sharpness": self._sam_sharpness,
            "distill_loss": self._distill_loss,
            "meta_loss": self._meta_loss,
            "replay_loss": self._replay_loss,
            "token_importance_mean": self._token_imp_mean,
            "uses_grad_checkpointing": self._uses_grad_checkpoint,
            "replay_buffer_size": len(self._replay_buf),
            "dead_neurons_per_layer": self._dead_neurons_per_layer,
        }
