"""
enhanced/metrics.py — Compute all 48 diagnostic metrics from hook snapshots + optimizer state.

24 original AI metrics + 24 new metrics covering perplexity, calibration, Fisher information,
SAM sharpness, meta-gradient alignment, distillation, replay, LoRA rank utilization, etc.

Also extends BioMetrics to accommodate the new bio signals from enhanced bio_model.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import torch


# ---------------------------------------------------------------------------
# AIMetrics — 48 fields
# ---------------------------------------------------------------------------

@dataclass
class AIMetrics:
    """All 48 AI-side diagnostic values from one training step."""
    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0

    # ---- Original 24 metrics ----
    # 1. Connectivity (mean weight norm)
    connectivity: float = 0.0
    # 2. Learning curve (cumulative loss trend)
    loss_curve: float = 0.0
    # 3. Error signal (loss delta)
    error_delta: float = 0.0
    # 4. Mutual information proxy (mean activation entropy)
    mutual_info: float = 0.0
    # 5. Gradient norms per layer
    gradient_norms: list = field(default_factory=list)
    # 6. Latent space (PCA of hidden states)
    latent_x: list = field(default_factory=list)
    latent_y: list = field(default_factory=list)
    latent_labels: list = field(default_factory=list)
    # 7. Activation survival per layer
    activation_survival: list = field(default_factory=list)
    # 8. Attention entropy per head
    attention_entropy: list = field(default_factory=list)
    mean_attention_entropy: float = 0.0
    # 9. Weight norms per layer
    weight_norms: list = field(default_factory=list)
    # 10. Loss landscape sharpness
    loss_sharpness: float = 0.0
    # 11. LayerNorm stats
    ln_means: list = field(default_factory=list)
    ln_vars: list = field(default_factory=list)
    # 12. Confidence
    confidence: float = 0.0
    prediction_entropy: float = 0.0
    # 13. Momentum velocity
    momentum_velocity: float = 0.0
    # 14. Effective learning rate
    effective_lr: float = 0.0
    # 15. Weight sparsity
    weight_sparsity: list = field(default_factory=list)
    mean_sparsity: float = 0.0
    # 16. Residual stream ratio
    residual_ratios: list = field(default_factory=list)
    # 17. Embedding cosine drift
    embedding_drift: float = 0.0
    # 18. Context utilisation
    context_mass: list = field(default_factory=list)
    effective_context_ratio: float = 0.0
    # 19. Gradient clipping
    raw_grad_norm: float = 0.0
    clip_triggered: bool = False
    clip_rate: float = 0.0
    # 20. Forgetting score
    forgetting_score: float = 0.0
    # 21. Effective rank
    effective_rank: float = 0.0
    effective_rank_normalized: float = 0.0
    # 22. Gradient SNR
    gradient_snr: float = 0.0
    # 23. Cross-layer CKA
    cka_values: list = field(default_factory=list)
    mean_cka: float = 0.0
    # 24. Expert routing entropy
    expert_load: list = field(default_factory=list)
    expert_entropy: float = 0.0

    # ---- New 24 metrics ----
    # 25. Perplexity
    perplexity: float = 1.0
    # 26. Prediction calibration (ECE)
    prediction_calibration: float = 0.0
    # 27. Attention head entropy
    attention_head_entropy: float = 0.0
    # 28. Dead head fraction
    dead_head_fraction: float = 0.0
    # 29. Gradient direction coherence
    gradient_direction_coherence: float = 0.0
    # 30. Weight update velocity mean
    weight_update_velocity_mean: float = 0.0
    # 31. Fisher-weighted loss
    fisher_weighted_loss: float = 0.0
    # 32. Activation kurtosis mean
    activation_kurtosis_mean: float = 0.0
    # 33. Spectral radius
    spectral_radius: float = 0.0
    # 34. Hessian trace estimate (Hutchinson)
    hessian_trace_estimate: float = 0.0
    # 35. Layer saturation mean
    layer_saturation_mean: float = 0.0
    # 36. Feature redundancy mean
    feature_redundancy_mean: float = 0.0
    # 37. Token perplexity variance
    token_perplexity_variance: float = 0.0
    # 38. Curriculum difficulty
    curriculum_difficulty: float = 0.0
    # 39. Adaptation speed
    adaptation_speed: float = 0.0
    # 40. Knowledge retention
    knowledge_retention: float = 1.0
    # 41. Parameter efficiency
    parameter_efficiency: float = 0.0
    # 42. Generalization gap estimate
    generalization_gap_estimate: float = 0.0
    # 43. SAM sharpness
    sam_sharpness: float = 0.0
    # 44. Meta-gradient alignment
    meta_gradient_alignment: float = 0.0
    # 45. Replay benefit
    replay_benefit: float = 0.0
    # 46. Distillation agreement
    distillation_agreement: float = 0.0
    # 47. LoRA rank utilization
    lora_rank_utilization: float = 0.0
    # 48. Cross-layer information flow
    cross_layer_information_flow: float = 0.0


# ---------------------------------------------------------------------------
# BioMetrics — original 24 + new extended fields
# ---------------------------------------------------------------------------

@dataclass
class BioMetrics:
    """All bio-side diagnostic values including the extended set."""
    step: int = 0

    # --- Original 24 ---
    synaptic_weight: float = 0.0
    mastery: float = 0.0
    delta: float = 0.0
    dopamine: float = 0.0
    sparsity: float = 0.0
    dendritic_signal: list = field(default_factory=list)
    place_x: list = field(default_factory=list)
    place_y: list = field(default_factory=list)
    place_labels: list = field(default_factory=list)
    active_fraction: float = 0.0
    attention_focus: float = 0.0
    homeostatic_norms: list = field(default_factory=list)
    attractor_variance: float = 0.0
    mean_firing_rate: float = 0.0
    certainty: float = 0.0
    habit_strength: float = 0.0
    plasticity: float = 0.0
    synapse_density: float = 0.0
    bypass_ratios: list = field(default_factory=list)
    cortical_stability: float = 0.0
    wm_profile: list = field(default_factory=list)
    pain_triggered: float = 0.0
    pain_rate: float = 0.0
    memory_retention: float = 0.0
    population_diversity: float = 0.0
    noise_benefit: float = 0.0
    column_differentiation: float = 0.0
    specialisation: list = field(default_factory=list)
    specialisation_entropy: float = 0.0

    # Visualization helpers (original)
    neuron_activities: list = field(default_factory=list)
    spike_pairs: list = field(default_factory=list)

    # --- Extended bio fields from enhanced bio_model ---
    # Oscillation band powers
    oscillation_bands: dict = field(default_factory=lambda: {
        "delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0
    })
    # Brain state
    brain_state: str = "resting"
    # Thalamic input level
    thalamic_input: float = 0.0
    # Neuromodulator levels
    neuromodulator_levels: dict = field(default_factory=lambda: {
        "dopamine": 0.5, "acetylcholine": 0.5, "norepinephrine": 0.5
    })
    # Column synchrony matrix (10x10 pairwise)
    column_sync: list = field(default_factory=list)
    # Apical prediction error
    apical_error: float = 0.0
    # Memory consolidation score
    consolidation_score: float = 0.0
    # 3D brain positions for visualization
    brain_positions: list = field(default_factory=list)
    # Spike arc paths for visualization
    spike_arcs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# MetricComputer
# ---------------------------------------------------------------------------

class MetricComputer:
    """Computes all 48 AI-side metrics from hook snapshots and optimizer state."""

    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self._loss_history: List[float] = []
        self._grad_norm_history: List[float] = []
        self._clip_history: List[float] = []
        self._initial_embedding: Optional[torch.Tensor] = None
        self._pruning_threshold = config.get("training", {}).get("pruning_threshold", 0.05)
        self._prev_grad_vec: Optional[np.ndarray] = None
        self._prev_weight_norms: Optional[List[float]] = None
        self._adaptation_loss_history: List[float] = []

    def set_initial_embedding(self, emb_tensor: torch.Tensor):
        self._initial_embedding = emb_tensor.detach().float().cpu()

    def compute(
        self,
        hook_snapshot,
        optimizer,
        scheduler,
        loss_value: float,
        step: int,
        raw_grad_norm: float,
        clip_triggered: bool,
        logits: Optional[torch.Tensor] = None,
        forgetting_score: float = 0.0,
        # Extended kwargs (new metrics)
        perplexity: float = 1.0,
        dead_neurons_per_layer: Optional[List[float]] = None,
        sam_sharpness: float = 0.0,
        distill_loss: float = 0.0,
        meta_loss: float = 0.0,
        replay_loss: float = 0.0,
        token_importance_mean: float = 0.0,
    ) -> AIMetrics:
        m = AIMetrics(step=step, loss=loss_value)

        # ---- Loss tracking ----
        self._loss_history.append(loss_value)
        if len(self._loss_history) > 50:
            self._loss_history = self._loss_history[-50:]

        self._grad_norm_history.append(raw_grad_norm)
        if len(self._grad_norm_history) > 50:
            self._grad_norm_history = self._grad_norm_history[-50:]

        self._clip_history.append(1.0 if clip_triggered else 0.0)
        if len(self._clip_history) > 50:
            self._clip_history = self._clip_history[-50:]

        snap = hook_snapshot

        # ========== ORIGINAL 24 METRICS ==========

        # 1. Connectivity
        m.weight_norms = snap.layer_weight_norms
        m.connectivity = float(np.mean(snap.layer_weight_norms)) if snap.layer_weight_norms else 0.0

        # 2. Loss curve
        m.loss_curve = loss_value

        # 3. Error signal
        if len(self._loss_history) >= 2:
            m.error_delta = self._loss_history[-1] - self._loss_history[-2]
        else:
            m.error_delta = 0.0

        # 4. Mutual information proxy
        if snap.layer_activation_norms:
            vals = [v for v in snap.layer_activation_norms if v > 0]
            m.mutual_info = float(np.mean(vals)) if vals else 0.0

        # 5. Gradient norms
        m.gradient_norms = snap.layer_gradient_norms

        # 6. Latent space (PCA)
        if snap.hidden_vectors and len(snap.hidden_vectors) >= 2:
            try:
                mat = np.array(snap.hidden_vectors, dtype=np.float32)
                mat = mat - mat.mean(axis=0)
                U, S, Vt = np.linalg.svd(mat, full_matrices=False)
                coords = U[:, :2] * S[:2]
                m.latent_x = coords[:, 0].tolist()
                m.latent_y = coords[:, 1].tolist() if coords.shape[1] > 1 else [0.0] * len(m.latent_x)
                n_pts = len(m.latent_x)
                m.latent_labels = (
                    [0] * (n_pts // 3) + [1] * (n_pts // 3) + [2] * (n_pts - 2 * (n_pts // 3))
                )
            except Exception:
                pass

        # 7. Activation survival
        m.activation_survival = snap.activation_survival

        # 8. Attention entropy
        all_ent = []
        for layer_ent in snap.attention_entropies:
            if layer_ent:
                all_ent.extend(layer_ent)
        m.attention_entropy = all_ent
        m.mean_attention_entropy = float(np.mean(all_ent)) if all_ent else 0.0

        # 9. Weight norms (already set)

        # 10. Loss sharpness
        if len(self._loss_history) >= 5:
            m.loss_sharpness = float(np.var(self._loss_history[-20:]))
        else:
            m.loss_sharpness = 0.0

        # 11. LayerNorm stats
        m.ln_means = [s[0] for s in snap.layernorm_stats]
        m.ln_vars = [s[1] for s in snap.layernorm_stats]

        # 12. Confidence from logits
        if logits is not None:
            try:
                with torch.no_grad():
                    probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
                    log_probs = torch.log(probs + 1e-10)
                    ent = -(probs * log_probs).sum(dim=-1).mean().item()
                    max_ent = math.log(logits.shape[-1])
                    m.prediction_entropy = ent
                    m.confidence = max(0.0, 1.0 - ent / max_ent)
            except Exception:
                m.confidence = 0.0
                m.prediction_entropy = 0.0

        # 13. Momentum velocity
        try:
            velocities = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        state = optimizer.state.get(p)
                        if state and "exp_avg" in state:
                            velocities.append(state["exp_avg"].detach().float().norm().item())
            m.momentum_velocity = float(np.mean(velocities)) if velocities else 0.0
        except Exception:
            m.momentum_velocity = 0.0

        # 14. Effective LR
        try:
            m.effective_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
        except Exception:
            m.effective_lr = optimizer.param_groups[0]["lr"]

        # 15. Weight sparsity
        sparsities = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and "lora" in name:
                try:
                    s = (param.detach().abs() < self._pruning_threshold).float().mean().item()
                    sparsities.append(s)
                except Exception:
                    pass
        m.weight_sparsity = sparsities
        m.mean_sparsity = float(np.mean(sparsities)) if sparsities else 0.0

        # 16. Residual ratios
        m.residual_ratios = snap.residual_ratios

        # 17. Embedding drift
        if self._initial_embedding is not None:
            try:
                current_emb = None
                for name, param in self.model.named_parameters():
                    if "embed_tokens" in name and "weight" in name:
                        current_emb = param.detach().float().cpu()
                        break
                if current_emb is not None:
                    flat_now = current_emb.flatten()
                    flat_init = self._initial_embedding.flatten()
                    cos_sim = torch.dot(flat_now, flat_init) / (
                        flat_now.norm() * flat_init.norm() + 1e-10
                    )
                    m.embedding_drift = 1.0 - cos_sim.item()
            except Exception:
                m.embedding_drift = 0.0

        # 18. Context utilisation
        m.context_mass = snap.attention_position_mass
        if m.context_mass:
            half = len(m.context_mass) // 2
            recent_mass = sum(m.context_mass[half:])
            m.effective_context_ratio = recent_mass

        # 19. Gradient clipping
        m.raw_grad_norm = raw_grad_norm
        m.clip_triggered = clip_triggered
        m.clip_rate = float(np.mean(self._clip_history[-50:])) if self._clip_history else 0.0

        # 20. Forgetting
        m.forgetting_score = forgetting_score

        # 21. Effective rank
        try:
            all_weights = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and "lora" in name and param.dim() >= 2:
                    all_weights.append(param.detach().float().cpu())
            if all_weights:
                w = max(all_weights, key=lambda x: x.numel())
                if w.dim() == 2:
                    s = torch.linalg.svdvals(w)
                    s_norm = s / (s.sum() + 1e-10)
                    ent = -(s_norm * torch.log(s_norm + 1e-10)).sum().item()
                    m.effective_rank = math.exp(ent)
                    m.effective_rank_normalized = m.effective_rank / min(w.shape)
        except Exception:
            m.effective_rank = 0.0

        # 22. Gradient SNR
        if len(self._grad_norm_history) >= 5:
            mean_g = float(np.mean(self._grad_norm_history[-20:]))
            std_g = float(np.std(self._grad_norm_history[-20:])) + 1e-10
            m.gradient_snr = mean_g / std_g

        # 23. Cross-layer CKA
        cka_vals = []
        norms = snap.layer_activation_norms
        if len(norms) >= 2:
            for i in range(len(norms) - 1):
                if norms[i] > 0 and norms[i + 1] > 0:
                    ratio = min(norms[i], norms[i + 1]) / (max(norms[i], norms[i + 1]) + 1e-10)
                    cka_vals.append(ratio)
        m.cka_values = cka_vals
        m.mean_cka = float(np.mean(cka_vals)) if cka_vals else 0.0

        # 24. Expert routing
        if m.attention_entropy:
            total = sum(abs(e) for e in m.attention_entropy) + 1e-10
            loads = [abs(e) / total for e in m.attention_entropy]
            m.expert_load = loads
            m.expert_entropy = -sum(p * math.log(p + 1e-10) for p in loads) if loads else 0.0

        # ========== NEW 24 METRICS ==========

        # 25. Perplexity
        m.perplexity = perplexity if perplexity > 0 else math.exp(min(loss_value, 20.0))

        # 26. Prediction calibration (ECE proxy using confidence vs loss)
        try:
            if logits is not None:
                with torch.no_grad():
                    probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
                    max_prob = probs.max(dim=-1)[0].mean().item()
                    accuracy_proxy = max(0.0, 1.0 - loss_value / 10.0)
                    m.prediction_calibration = 1.0 - abs(max_prob - accuracy_proxy)
            else:
                m.prediction_calibration = 0.5
        except Exception:
            m.prediction_calibration = 0.5

        # 27. Attention head entropy (distribution across heads)
        if all_ent:
            ent_arr = np.array(all_ent)
            ent_arr = np.abs(ent_arr) + 1e-10
            ent_norm = ent_arr / ent_arr.sum()
            m.attention_head_entropy = float(-np.sum(ent_norm * np.log(ent_norm)))
        else:
            m.attention_head_entropy = 0.0

        # 28. Dead head fraction (attention heads with near-zero variance)
        dead_heads = 0
        total_heads = 0
        for layer_ent in snap.attention_entropies:
            for e in layer_ent:
                total_heads += 1
                if abs(e) < 1e-4:
                    dead_heads += 1
        m.dead_head_fraction = dead_heads / max(1, total_heads)

        # 29. Gradient direction coherence (cosine sim with previous gradient)
        try:
            grad_vec = np.array(snap.layer_gradient_norms, dtype=np.float32)
            if self._prev_grad_vec is not None and len(grad_vec) == len(self._prev_grad_vec):
                dot = np.dot(grad_vec, self._prev_grad_vec)
                norm = (np.linalg.norm(grad_vec) * np.linalg.norm(self._prev_grad_vec)) + 1e-10
                m.gradient_direction_coherence = float(np.clip(dot / norm, -1.0, 1.0))
            else:
                m.gradient_direction_coherence = 0.0
            self._prev_grad_vec = grad_vec.copy()
        except Exception:
            m.gradient_direction_coherence = 0.0

        # 30. Weight update velocity mean (||W_t - W_{t-1}||)
        try:
            curr_weight_norms = snap.layer_weight_norms
            if self._prev_weight_norms is not None and len(curr_weight_norms) == len(self._prev_weight_norms):
                velocities_wt = [
                    abs(c - p) for c, p in zip(curr_weight_norms, self._prev_weight_norms)
                ]
                m.weight_update_velocity_mean = float(np.mean(velocities_wt))
            else:
                m.weight_update_velocity_mean = 0.0
            self._prev_weight_norms = list(curr_weight_norms)
        except Exception:
            m.weight_update_velocity_mean = 0.0

        # 31. Fisher-weighted loss
        try:
            fisher_sum = 0.0
            fisher_count = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad and hasattr(snap, "fisher_diagonal_mean"):
                    fisher_sum += snap.fisher_diagonal_mean[fisher_count] if fisher_count < len(snap.fisher_diagonal_mean) else 0.0
                    fisher_count += 1
            fisher_mean = fisher_sum / max(1, fisher_count)
            m.fisher_weighted_loss = loss_value * (1.0 + fisher_mean)
        except Exception:
            m.fisher_weighted_loss = loss_value

        # 32. Activation kurtosis mean
        try:
            if hasattr(snap, "activation_histograms") and snap.activation_histograms:
                kurtoses = []
                for hist in snap.activation_histograms:
                    if hist and len(hist) >= 4:
                        arr = np.array(hist, dtype=np.float32)
                        arr = arr / (arr.sum() + 1e-10)
                        # approximate kurtosis from histogram
                        mean = np.average(np.arange(len(arr)), weights=arr)
                        var = np.average((np.arange(len(arr)) - mean) ** 2, weights=arr)
                        kurt = np.average((np.arange(len(arr)) - mean) ** 4, weights=arr) / (var ** 2 + 1e-10)
                        kurtoses.append(float(kurt))
                m.activation_kurtosis_mean = float(np.mean(kurtoses)) if kurtoses else 3.0
            else:
                m.activation_kurtosis_mean = 3.0  # Gaussian baseline
        except Exception:
            m.activation_kurtosis_mean = 3.0

        # 33. Spectral radius (largest singular value of largest LoRA weight)
        try:
            all_weights_lora = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and "lora" in name and param.dim() >= 2:
                    all_weights_lora.append(param.detach().float().cpu())
            if all_weights_lora:
                w = max(all_weights_lora, key=lambda x: x.numel())
                s = torch.linalg.svdvals(w)
                m.spectral_radius = float(s[0].item())
            else:
                m.spectral_radius = 0.0
        except Exception:
            m.spectral_radius = 0.0

        # 34. Hessian trace estimate (Hutchinson: E[v^T H v] where v ~ N(0,I))
        try:
            hessian_traces = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and "lora" in name:
                    v = torch.randn_like(param)
                    gv = (param.grad * v).sum()
                    hessian_traces.append(float(gv.item() ** 2))
            m.hessian_trace_estimate = float(np.mean(hessian_traces)) if hessian_traces else 0.0
        except Exception:
            m.hessian_trace_estimate = 0.0

        # 35. Layer saturation mean
        try:
            if hasattr(snap, "saturation_fractions") and snap.saturation_fractions:
                m.layer_saturation_mean = float(np.mean(snap.saturation_fractions))
            else:
                # Estimate from activation norms being too high
                if snap.layer_activation_norms:
                    max_norm = max(snap.layer_activation_norms) + 1e-10
                    saturations = [min(1.0, n / max_norm) for n in snap.layer_activation_norms]
                    m.layer_saturation_mean = float(np.mean(saturations))
                else:
                    m.layer_saturation_mean = 0.0
        except Exception:
            m.layer_saturation_mean = 0.0

        # 36. Feature redundancy mean
        try:
            if hasattr(snap, "feature_correlation_mean") and snap.feature_correlation_mean:
                m.feature_redundancy_mean = float(np.mean(snap.feature_correlation_mean))
            else:
                m.feature_redundancy_mean = 0.0
        except Exception:
            m.feature_redundancy_mean = 0.0

        # 37. Token perplexity variance
        try:
            if logits is not None:
                with torch.no_grad():
                    shift_logits = logits[:, :-1, :].float()
                    if shift_logits.shape[1] > 0:
                        # Use softmax entropy as proxy for per-token perplexity variance
                        token_ents = []
                        for t in range(min(shift_logits.shape[1], 32)):
                            probs_t = torch.softmax(shift_logits[:, t, :], dim=-1)
                            e_t = -(probs_t * torch.log(probs_t + 1e-10)).sum(dim=-1).mean()
                            token_ents.append(math.exp(min(e_t.item(), 10.0)))
                        m.token_perplexity_variance = float(np.var(token_ents))
                    else:
                        m.token_perplexity_variance = 0.0
            else:
                m.token_perplexity_variance = 0.0
        except Exception:
            m.token_perplexity_variance = 0.0

        # 38. Curriculum difficulty
        # Estimate difficulty as normalized loss relative to recent average
        if len(self._loss_history) >= 5:
            avg_recent = float(np.mean(self._loss_history[-10:]))
            m.curriculum_difficulty = float(np.clip(loss_value / (avg_recent + 1e-10), 0.0, 3.0))
        else:
            m.curriculum_difficulty = 1.0

        # 39. Adaptation speed (rate of loss decrease per gradient step)
        self._adaptation_loss_history.append(loss_value)
        if len(self._adaptation_loss_history) > 20:
            self._adaptation_loss_history = self._adaptation_loss_history[-20:]
        if len(self._adaptation_loss_history) >= 5:
            xs = np.arange(len(self._adaptation_loss_history), dtype=np.float32)
            ys = np.array(self._adaptation_loss_history, dtype=np.float32)
            # Linear regression slope
            slope = float(np.polyfit(xs, ys, 1)[0])
            m.adaptation_speed = -slope  # Positive = loss decreasing = faster adaptation
        else:
            m.adaptation_speed = 0.0

        # 40. Knowledge retention
        m.knowledge_retention = max(0.0, 1.0 - forgetting_score)

        # 41. Parameter efficiency (trainable_params / effective_rank)
        try:
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            eff_rank = max(1.0, m.effective_rank)
            m.parameter_efficiency = float(eff_rank / (n_trainable / 1e6 + 1e-10))
        except Exception:
            m.parameter_efficiency = 0.0

        # 42. Generalization gap estimate (train vs eval-mode loss proxy)
        # Use loss sharpness as a proxy: high sharpness suggests poor generalization
        m.generalization_gap_estimate = float(np.clip(m.loss_sharpness * 2.0, 0.0, 1.0))

        # 43. SAM sharpness (from trainer)
        m.sam_sharpness = sam_sharpness

        # 44. Meta-gradient alignment
        # Proxy: coherence of gradient direction after meta-update (use gradient_direction_coherence)
        m.meta_gradient_alignment = m.gradient_direction_coherence * (1.0 if meta_loss > 0 else 0.0)

        # 45. Replay benefit (how much replay reduces loss vs current loss)
        if replay_loss > 0 and loss_value > 0:
            m.replay_benefit = max(0.0, 1.0 - replay_loss / (loss_value + 1e-10))
        else:
            m.replay_benefit = 0.0

        # 46. Distillation agreement (KL div stored as distill_loss; lower = more agreement)
        m.distillation_agreement = float(np.clip(1.0 / (1.0 + distill_loss), 0.0, 1.0))

        # 47. LoRA rank utilization
        try:
            lora_rank_utils = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and "lora" in name and param.dim() == 2:
                    s = torch.linalg.svdvals(param.detach().float().cpu())
                    # Fraction of singular values above threshold
                    threshold = s.max() * 0.01
                    active = (s > threshold).float().mean().item()
                    lora_rank_utils.append(active)
            m.lora_rank_utilization = float(np.mean(lora_rank_utils)) if lora_rank_utils else 0.0
        except Exception:
            m.lora_rank_utilization = 0.0

        # 48. Cross-layer information flow (mutual information proxy between adjacent layers)
        try:
            flows = []
            norms = snap.layer_activation_norms
            if len(norms) >= 2:
                for i in range(len(norms) - 1):
                    n_i = norms[i] + 1e-10
                    n_j = norms[i + 1] + 1e-10
                    # Normalized mutual information proxy: min(n_i, n_j) / max(n_i, n_j)
                    flows.append(min(n_i, n_j) / max(n_i, n_j))
            m.cross_layer_information_flow = float(np.mean(flows)) if flows else 0.0
        except Exception:
            m.cross_layer_information_flow = 0.0

        return m

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self, m: AIMetrics) -> dict:
        """Serialize AIMetrics to a JSON-compatible dict (all 48 metrics)."""
        return {
            # --- Identity ---
            "step": m.step,
            "loss": m.loss,
            "accuracy": m.accuracy,
            # --- Original 24 ---
            "connectivity": m.connectivity,
            "loss_curve": m.loss_curve,
            "error_delta": m.error_delta,
            "mutual_info": m.mutual_info,
            "gradient_norms": m.gradient_norms,
            "latent_x": m.latent_x,
            "latent_y": m.latent_y,
            "latent_labels": m.latent_labels,
            "activation_survival": m.activation_survival,
            "attention_entropy": m.attention_entropy,
            "mean_attention_entropy": m.mean_attention_entropy,
            "weight_norms": m.weight_norms,
            "loss_sharpness": m.loss_sharpness,
            "ln_means": m.ln_means,
            "ln_vars": m.ln_vars,
            "confidence": m.confidence,
            "prediction_entropy": m.prediction_entropy,
            "momentum_velocity": m.momentum_velocity,
            "effective_lr": m.effective_lr,
            "weight_sparsity": m.weight_sparsity,
            "mean_sparsity": m.mean_sparsity,
            "residual_ratios": m.residual_ratios,
            "embedding_drift": m.embedding_drift,
            "context_mass": m.context_mass,
            "effective_context_ratio": m.effective_context_ratio,
            "raw_grad_norm": m.raw_grad_norm,
            "clip_triggered": m.clip_triggered,
            "clip_rate": m.clip_rate,
            "forgetting_score": m.forgetting_score,
            "effective_rank": m.effective_rank,
            "effective_rank_normalized": m.effective_rank_normalized,
            "gradient_snr": m.gradient_snr,
            "cka_values": m.cka_values,
            "mean_cka": m.mean_cka,
            "expert_load": m.expert_load,
            "expert_entropy": m.expert_entropy,
            # --- New 24 ---
            "perplexity": m.perplexity,
            "prediction_calibration": m.prediction_calibration,
            "attention_head_entropy": m.attention_head_entropy,
            "dead_head_fraction": m.dead_head_fraction,
            "gradient_direction_coherence": m.gradient_direction_coherence,
            "weight_update_velocity_mean": m.weight_update_velocity_mean,
            "fisher_weighted_loss": m.fisher_weighted_loss,
            "activation_kurtosis_mean": m.activation_kurtosis_mean,
            "spectral_radius": m.spectral_radius,
            "hessian_trace_estimate": m.hessian_trace_estimate,
            "layer_saturation_mean": m.layer_saturation_mean,
            "feature_redundancy_mean": m.feature_redundancy_mean,
            "token_perplexity_variance": m.token_perplexity_variance,
            "curriculum_difficulty": m.curriculum_difficulty,
            "adaptation_speed": m.adaptation_speed,
            "knowledge_retention": m.knowledge_retention,
            "parameter_efficiency": m.parameter_efficiency,
            "generalization_gap_estimate": m.generalization_gap_estimate,
            "sam_sharpness": m.sam_sharpness,
            "meta_gradient_alignment": m.meta_gradient_alignment,
            "replay_benefit": m.replay_benefit,
            "distillation_agreement": m.distillation_agreement,
            "lora_rank_utilization": m.lora_rank_utilization,
            "cross_layer_information_flow": m.cross_layer_information_flow,
        }

    def bio_to_dict(self, m: BioMetrics) -> dict:
        """Serialize BioMetrics to a JSON-compatible dict (all fields)."""
        return {
            "step": m.step,
            # --- Original 24 ---
            "synaptic_weight": m.synaptic_weight,
            "mastery": m.mastery,
            "delta": m.delta,
            "dopamine": m.dopamine,
            "sparsity": m.sparsity,
            "dendritic_signal": m.dendritic_signal,
            "place_x": m.place_x,
            "place_y": m.place_y,
            "place_labels": m.place_labels,
            "active_fraction": m.active_fraction,
            "attention_focus": m.attention_focus,
            "homeostatic_norms": m.homeostatic_norms,
            "attractor_variance": m.attractor_variance,
            "mean_firing_rate": m.mean_firing_rate,
            "certainty": m.certainty,
            "habit_strength": m.habit_strength,
            "plasticity": m.plasticity,
            "synapse_density": m.synapse_density,
            "bypass_ratios": m.bypass_ratios,
            "cortical_stability": m.cortical_stability,
            "wm_profile": m.wm_profile,
            "pain_triggered": m.pain_triggered,
            "pain_rate": m.pain_rate,
            "memory_retention": m.memory_retention,
            "population_diversity": m.population_diversity,
            "noise_benefit": m.noise_benefit,
            "column_differentiation": m.column_differentiation,
            "specialisation": m.specialisation,
            "specialisation_entropy": m.specialisation_entropy,
            # Visualization
            "neuron_activities": m.neuron_activities,
            "spike_pairs": m.spike_pairs,
            # --- Extended bio fields ---
            "oscillation_bands": m.oscillation_bands,
            "brain_state": m.brain_state,
            "thalamic_input": m.thalamic_input,
            "neuromodulator_levels": m.neuromodulator_levels,
            "column_sync": m.column_sync,
            "apical_error": m.apical_error,
            "consolidation_score": m.consolidation_score,
            "brain_positions": m.brain_positions,
            "spike_arcs": m.spike_arcs,
        }
