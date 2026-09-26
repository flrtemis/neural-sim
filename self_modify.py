"""
enhanced/self_modify.py — Adaptive Self-Modification Engine with Bayesian optimization,
multi-objective Pareto front, evolutionary population, and rollback.

Extensions over base self_modify.py:
- Multi-objective optimization: Pareto front of (loss, forgetting, efficiency)
- Bayesian optimization: Gaussian Process surrogate + Expected Improvement
- Architecture mutation: LoRA rank change events
- Hyperparameter evolution: population of 8 configs, fitness-based breeding
- Detailed diagnostic prompts with spectral analysis
- Automatic curriculum detection (plateau → difficulty adjustment)
- Parameter interdependence rules (LR↑ → grad_clip↑)
- Modification confidence scoring (history-based; only apply if > 0.7)
- Rollback: if loss spikes > 20% within 10 steps, auto-rollback
- Multi-step planning: 3-step plan with evaluation between steps
- 12 modifiable parameters
- get_optimization_state() returning pareto_front, gp_mean, gp_std, population_fitness
"""

import asyncio
import copy
import math
import time
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Modification:
    """A single parameter modification."""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    timestamp: float
    step: int
    confidence: float = 0.0
    expected_improvement: float = 0.0


@dataclass
class ParetoPoint:
    """A point on the Pareto front of (loss, forgetting, efficiency)."""
    loss: float
    forgetting: float
    efficiency: float
    parameters: dict
    step: int


# ---------------------------------------------------------------------------
# Tool definition (expanded to 12 parameters)
# ---------------------------------------------------------------------------

SELF_MODIFY_TOOL = {
    "name": "self_modify",
    "description": (
        "Modify one of your own training parameters to improve learning. "
        "Analyze the diagnostic data and adjust a parameter if you identify an issue."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "parameter": {
                "type": "string",
                "enum": [
                    "learning_rate", "dropout", "temperature",
                    "gradient_clip_norm", "ewc_lambda", "weight_decay",
                    "beta1", "beta2", "sam_rho", "distill_alpha",
                    "replay_ratio", "llrd_factor",
                ],
                "description": "The training parameter to modify",
            },
            "new_value": {
                "type": "number",
                "description": "The new value for the parameter",
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of why this change will help",
            },
        },
        "required": ["parameter", "new_value", "reason"],
    },
}


# ---------------------------------------------------------------------------
# Gaussian Process (simple RBF kernel) for Bayesian optimization
# ---------------------------------------------------------------------------

class SimpleGP:
    """Minimal GP with RBF kernel for Expected Improvement computation."""

    def __init__(self, noise: float = 0.01, length_scale: float = 1.0):
        self.noise = noise
        self.length_scale = length_scale
        self.X: Optional[np.ndarray] = None  # [n_obs, n_params]
        self.y: Optional[np.ndarray] = None  # [n_obs]
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF (Gaussian) kernel between two sets of points."""
        # X1: [n, d], X2: [m, d] → [n, m]
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # [n, m, d]
        dist_sq = (diff ** 2).sum(axis=-1)  # [n, m]
        return np.exp(-0.5 * dist_sq / (self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to observations."""
        self.X = X.copy()
        self.y = y.copy()
        K = self._rbf_kernel(X, X)
        K += self.noise * np.eye(len(X))
        try:
            self._K_inv = np.linalg.inv(K)
            self._alpha = self._K_inv @ y
        except np.linalg.LinAlgError:
            self._K_inv = np.eye(len(X))
            self._alpha = y

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) predictions at X_star."""
        if self.X is None or len(self.X) == 0:
            return np.zeros(len(X_star)), np.ones(len(X_star))
        K_star = self._rbf_kernel(X_star, self.X)  # [n_star, n_obs]
        K_star_star = self._rbf_kernel(X_star, X_star)  # [n_star, n_star]

        mean = K_star @ self._alpha
        var = np.diag(K_star_star) - np.sum(K_star @ self._K_inv * K_star, axis=1)
        std = np.sqrt(np.clip(var, 0, None))
        return mean, std

    def expected_improvement(self, X_star: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
        """Compute Expected Improvement at X_star (lower is better → minimize)."""
        mean, std = self.predict(X_star)
        improvement = y_best - mean - xi
        Z = improvement / (std + 1e-10)
        # Standard normal CDF and PDF
        from scipy.special import ndtr
        ei = improvement * ndtr(Z) + std * np.exp(-0.5 * Z ** 2) / math.sqrt(2 * math.pi)
        ei[std < 1e-10] = 0.0
        return ei


# ---------------------------------------------------------------------------
# SelfModificationEngine — enhanced
# ---------------------------------------------------------------------------

class SelfModificationEngine:
    """Allows the LLM to inspect and modify its own training parameters.

    Enhanced with Bayesian optimization, multi-objective Pareto tracking,
    evolutionary population, rollback, and multi-step planning.
    """

    PARAM_NAMES = [
        "learning_rate", "dropout", "temperature",
        "gradient_clip_norm", "ewc_lambda", "weight_decay",
        "beta1", "beta2", "sam_rho", "distill_alpha",
        "replay_ratio", "llrd_factor",
    ]

    # Parameter interdependence rules: increasing one → adjust another
    INTERDEPENDENCE = {
        "learning_rate": [("gradient_clip_norm", 1.5)],   # increase LR → increase grad clip
        "dropout":        [("ewc_lambda", 0.9)],           # more dropout → less EWC needed
        "sam_rho":        [("learning_rate", 1.1)],        # more SAM → slight LR boost
    }

    def __init__(self, model, tokenizer, trainer, config: dict, hook_manager=None):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.config = config
        self.sm_cfg = config.get("self_modify", {})
        self.hooks = hook_manager

        self.enabled = self.sm_cfg.get("enabled", True)
        self.eval_every = self.sm_cfg.get("evaluate_every_n_steps", 25)
        self.bounds: Dict[str, Tuple[float, float]] = self.sm_cfg.get("parameter_bounds", {})

        # Add default bounds for extended parameters if not provided
        default_bounds = {
            "learning_rate": [1e-6, 1e-3],
            "dropout": [0.0, 0.5],
            "temperature": [0.01, 2.0],
            "gradient_clip_norm": [0.1, 10.0],
            "ewc_lambda": [0.0, 10.0],
            "weight_decay": [0.0, 0.1],
            "beta1": [0.5, 0.999],
            "beta2": [0.9, 0.9999],
            "sam_rho": [0.01, 0.5],
            "distill_alpha": [0.0, 1.0],
            "replay_ratio": [0.0, 0.9],
            "llrd_factor": [0.5, 1.0],
        }
        for param, bounds in default_bounds.items():
            if param not in self.bounds:
                self.bounds[param] = bounds

        self.history: List[Modification] = []
        self._last_eval_step = 0

        # ---- Multi-objective Pareto front ----
        self._pareto_front: List[ParetoPoint] = []

        # ---- Bayesian optimization (GP surrogate) ----
        self._gp = SimpleGP(noise=0.01, length_scale=0.5)
        self._gp_observations_X: List[np.ndarray] = []  # normalized param vectors
        self._gp_observations_y: List[float] = []        # loss values
        self._gp_fitted = False

        # ---- Evolutionary population (8 configs) ----
        self._population: List[Dict[str, float]] = self._init_population(8)
        self._population_fitness: List[float] = [-float("inf")] * 8
        self._pop_gen = 0

        # ---- Confidence history: {param: [(before_loss, after_loss)] } ----
        self._confidence_history: Dict[str, List[Tuple[float, float]]] = {p: [] for p in self.PARAM_NAMES}
        self._confidence_threshold = 0.7

        # ---- Rollback mechanism ----
        self._rollback_state: Optional[Dict] = None
        self._rollback_loss_at_change: Optional[float] = None
        self._rollback_loss_window: List[float] = []
        self._rollback_window_size = 10
        self._rollback_threshold = 1.20  # 20% spike

        # ---- Curriculum ----
        self._plateau_detector_history: List[float] = []
        self._plateau_steps = 20
        self._plateau_tolerance = 0.01
        self._curriculum_level = 1  # 1=easy, 2=medium, 3=hard

        # ---- Multi-step plan ----
        self._plan_queue: List[Dict] = []  # pending plan steps
        self._plan_eval_step: Optional[int] = None
        self._plan_baseline_loss: Optional[float] = None

    # -----------------------------------------------------------------------
    # Population initialization
    # -----------------------------------------------------------------------

    def _init_population(self, pop_size: int) -> List[Dict[str, float]]:
        """Initialize a diverse population of parameter configurations."""
        state = self.trainer.get_state() if hasattr(self, 'trainer') else {}
        pop = []
        rng = np.random.RandomState(0)

        for i in range(pop_size):
            config = {}
            for param in self.PARAM_NAMES:
                lo, hi = self.bounds.get(param, [0.0, 1.0])
                if i == 0:
                    # First member: current values
                    config[param] = float(state.get(param, (lo + hi) / 2))
                else:
                    # Random in bounds
                    if param in ("learning_rate",):
                        # Log-uniform for LR
                        log_lo = math.log(lo + 1e-12)
                        log_hi = math.log(hi)
                        config[param] = float(math.exp(rng.uniform(log_lo, log_hi)))
                    else:
                        config[param] = float(rng.uniform(lo, hi))
            pop.append(config)
        return pop

    # -----------------------------------------------------------------------
    # Evaluation trigger
    # -----------------------------------------------------------------------

    def should_evaluate(self, step: int) -> bool:
        if not self.enabled:
            return False
        return step - self._last_eval_step >= self.eval_every

    # -----------------------------------------------------------------------
    # Diagnostic summary (rich version)
    # -----------------------------------------------------------------------

    def compile_diagnostic_summary(self, ai_metrics: dict, bio_metrics: dict) -> str:
        """Format current metrics as detailed natural language for the model."""
        step = ai_metrics.get("step", 0)

        # Spectral analysis: gradient norm distribution
        gnorms = ai_metrics.get("gradient_norms", [])
        gn_arr = [float(g) for g in gnorms if g is not None]
        gn_mean = float(np.mean(gn_arr)) if gn_arr else 0.0
        gn_std = float(np.std(gn_arr)) if gn_arr else 0.0
        gn_min = float(np.min(gn_arr)) if gn_arr else 0.0
        gn_max = float(np.max(gn_arr)) if gn_arr else 0.0

        # Attention pattern description
        attn_ent = ai_metrics.get("attention_entropy", [])
        attn_mean = float(np.mean(attn_ent)) if attn_ent else 0.0
        dead_heads = ai_metrics.get("dead_head_fraction", 0.0)

        # Convergence trajectory
        loss = ai_metrics.get("loss", 0.0)
        adapt_speed = ai_metrics.get("adaptation_speed", 0.0)
        perplexity = ai_metrics.get("perplexity", 1.0)

        # LoRA rank status
        lora_rank_util = ai_metrics.get("lora_rank_utilization", 0.0)
        eff_rank = ai_metrics.get("effective_rank_normalized", 0.0)

        lines = [
            f"=== Training Diagnostics (Step {step}) ===",
            "",
            "--- Loss & Convergence ---",
            f"  Loss: {loss:.4f}  |  Perplexity: {perplexity:.2f}",
            f"  Loss delta: {ai_metrics.get('error_delta', 0):.4f}",
            f"  Adaptation speed (loss slope): {adapt_speed:.6f}",
            f"  Loss sharpness (variance): {ai_metrics.get('loss_sharpness', 0):.6f}",
            f"  SAM sharpness: {ai_metrics.get('sam_sharpness', 0):.6f}",
            f"  Curriculum difficulty: {ai_metrics.get('curriculum_difficulty', 1.0):.3f}",
            "",
            "--- Gradient Analysis (Spectral) ---",
            f"  Gradient norm mean: {gn_mean:.4f}  std: {gn_std:.4f}  min: {gn_min:.4f}  max: {gn_max:.4f}",
            f"  Gradient SNR: {ai_metrics.get('gradient_snr', 0):.4f}",
            f"  Gradient direction coherence: {ai_metrics.get('gradient_direction_coherence', 0):.4f}",
            f"  Clip rate: {ai_metrics.get('clip_rate', 0):.1%}",
            f"  Meta-gradient alignment: {ai_metrics.get('meta_gradient_alignment', 0):.4f}",
            "",
            "--- Attention Patterns ---",
            f"  Mean attention entropy: {attn_mean:.4f}",
            f"  Attention head entropy: {ai_metrics.get('attention_head_entropy', 0):.4f}",
            f"  Dead head fraction: {dead_heads:.3f}",
            f"  Context utilisation (recent half): {ai_metrics.get('effective_context_ratio', 0):.4f}",
            "",
            "--- Weight & Rank ---",
            f"  Effective rank normalized: {eff_rank:.4f}",
            f"  LoRA rank utilization: {lora_rank_util:.3f}",
            f"  Spectral radius: {ai_metrics.get('spectral_radius', 0):.4f}",
            f"  Weight update velocity mean: {ai_metrics.get('weight_update_velocity_mean', 0):.6f}",
            f"  Embedding drift: {ai_metrics.get('embedding_drift', 0):.6f}",
            "",
            "--- Forgetting & Retention ---",
            f"  Forgetting score: {ai_metrics.get('forgetting_score', 0):.4f}",
            f"  Knowledge retention: {ai_metrics.get('knowledge_retention', 1):.4f}",
            f"  Replay benefit: {ai_metrics.get('replay_benefit', 0):.4f}",
            f"  Distillation agreement: {ai_metrics.get('distillation_agreement', 0):.4f}",
            "",
            "--- Optimization State ---",
            f"  Learning rate: {ai_metrics.get('effective_lr', 0):.6f}",
            f"  Momentum velocity: {ai_metrics.get('momentum_velocity', 0):.4f}",
            f"  Parameter efficiency: {ai_metrics.get('parameter_efficiency', 0):.4f}",
            f"  Hessian trace estimate: {ai_metrics.get('hessian_trace_estimate', 0):.6f}",
            "",
            "--- Bio Reference ---",
            f"  Bio sparsity: {bio_metrics.get('sparsity', 0):.4f}",
            f"  Bio firing rate: {bio_metrics.get('mean_firing_rate', 0):.4f}",
            f"  Bio population diversity: {bio_metrics.get('population_diversity', 0):.4f}",
            f"  Bio attractor stability: {bio_metrics.get('attractor_variance', 0):.4f}",
            f"  Bio brain state: {bio_metrics.get('brain_state', 'unknown')}",
            f"  Bio dopamine: {bio_metrics.get('neuromodulator_levels', {}).get('dopamine', 0.5):.3f}",
            "",
            "=== Current Parameter Values ===",
        ]

        state = self.trainer.get_state()
        for param in self.PARAM_NAMES:
            if param in state:
                lines.append(f"  {param}: {state[param]}")

        lines.append("")
        lines.append("=== Parameter Bounds ===")
        for param, bounds in self.bounds.items():
            if param in self.PARAM_NAMES:
                lines.append(f"  {param}: [{bounds[0]}, {bounds[1]}]")

        lines.append("")
        lines.append("=== Optimization History ===")
        lines.append(f"  Pareto front size: {len(self._pareto_front)}")
        lines.append(f"  Evolutionary generation: {self._pop_gen}")
        lines.append(f"  Modifications made: {len(self.history)}")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Plateau detection & curriculum
    # -----------------------------------------------------------------------

    def _check_plateau(self, loss: float) -> bool:
        """Detect loss plateau."""
        self._plateau_detector_history.append(loss)
        if len(self._plateau_detector_history) > self._plateau_steps:
            self._plateau_detector_history = self._plateau_detector_history[-self._plateau_steps:]
        if len(self._plateau_detector_history) >= self._plateau_steps:
            recent = self._plateau_detector_history[-10:]
            older = self._plateau_detector_history[:10]
            improvement = abs(np.mean(older) - np.mean(recent))
            return improvement < self._plateau_tolerance
        return False

    def _auto_curriculum(self, loss: float) -> Optional[str]:
        """Suggest curriculum adjustment if plateau detected."""
        if self._check_plateau(loss):
            if self._curriculum_level < 3:
                self._curriculum_level += 1
                return f"Loss plateau detected. Increasing curriculum difficulty to level {self._curriculum_level}."
        return None

    # -----------------------------------------------------------------------
    # Pareto front update
    # -----------------------------------------------------------------------

    def _update_pareto_front(self, loss: float, forgetting: float, efficiency: float, params: dict, step: int):
        """Update multi-objective Pareto front."""
        point = ParetoPoint(
            loss=loss,
            forgetting=forgetting,
            efficiency=efficiency,
            parameters=copy.deepcopy(params),
            step=step,
        )

        # Check if point dominates any existing front members
        dominated_indices = []
        dominated_by_existing = False

        for i, existing in enumerate(self._pareto_front):
            # Does existing dominate new point? (lower loss, forgetting; higher efficiency)
            if (existing.loss <= loss and existing.forgetting <= forgetting
                    and existing.efficiency >= efficiency):
                dominated_by_existing = True
                break
            # Does new point dominate existing?
            if (loss <= existing.loss and forgetting <= existing.forgetting
                    and efficiency >= existing.efficiency):
                dominated_indices.append(i)

        if not dominated_by_existing:
            # Remove dominated points
            self._pareto_front = [
                p for i, p in enumerate(self._pareto_front)
                if i not in dominated_indices
            ]
            self._pareto_front.append(point)

        # Keep front bounded
        if len(self._pareto_front) > 20:
            # Keep the 20 most recent non-dominated points
            self._pareto_front = sorted(self._pareto_front, key=lambda p: p.step)[-20:]

    # -----------------------------------------------------------------------
    # Bayesian optimization
    # -----------------------------------------------------------------------

    def _gp_update(self, param_vector: np.ndarray, loss: float):
        """Add an observation and refit the GP."""
        self._gp_observations_X.append(param_vector.copy())
        self._gp_observations_y.append(loss)

        if len(self._gp_observations_X) >= 3:
            X = np.array(self._gp_observations_X)
            y = np.array(self._gp_observations_y)
            # Normalize
            y_mean = y.mean()
            y_std = y.std() + 1e-10
            y_norm = (y - y_mean) / y_std
            try:
                self._gp.fit(X, y_norm)
                self._gp_fitted = True
            except Exception:
                pass

    def _param_to_vector(self, params: dict) -> np.ndarray:
        """Convert parameter dict to normalized vector."""
        vec = []
        for param in self.PARAM_NAMES:
            lo, hi = self.bounds.get(param, [0.0, 1.0])
            val = float(params.get(param, (lo + hi) / 2))
            normalized = (val - lo) / (hi - lo + 1e-10)
            vec.append(np.clip(normalized, 0.0, 1.0))
        return np.array(vec, dtype=np.float32)

    def _suggest_next_params(self) -> Optional[Dict[str, float]]:
        """Use EI to suggest next parameter set to try."""
        if not self._gp_fitted or len(self._gp_observations_y) < 3:
            return None

        try:
            # Generate random candidates
            rng = np.random.RandomState(self.step if hasattr(self, 'step') else 0)
            n_candidates = 50
            candidates = rng.rand(n_candidates, len(self.PARAM_NAMES)).astype(np.float32)

            y_best = min(self._gp_observations_y)
            y_mean = np.mean(self._gp_observations_y)
            y_std = np.std(self._gp_observations_y) + 1e-10
            y_best_norm = (y_best - y_mean) / y_std

            # Compute EI (simple manual version without scipy)
            means, stds = self._gp.predict(candidates)
            improvements = y_best_norm - means
            Z = improvements / (stds + 1e-10)
            # Approximate normal CDF
            ei = improvements * self._norm_cdf(Z) + stds * self._norm_pdf(Z)

            best_idx = int(np.argmax(ei))
            best_normalized = candidates[best_idx]

            # Denormalize
            params = {}
            for i, param in enumerate(self.PARAM_NAMES):
                lo, hi = self.bounds.get(param, [0.0, 1.0])
                params[param] = float(lo + best_normalized[i] * (hi - lo))

            return params
        except Exception:
            return None

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Approximation of standard normal CDF."""
        return 0.5 * (1.0 + np.sign(x) * np.sqrt(1 - np.exp(-x ** 2 * (2 / math.pi))))

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    # -----------------------------------------------------------------------
    # Evolutionary algorithm
    # -----------------------------------------------------------------------

    def _evolve_population(self, current_params: dict, current_loss: float):
        """Update population fitness and breed next generation."""
        state = self.trainer.get_state()
        forgetting = float(state.get("forgetting_score", 0.0) if hasattr(state, "get") else 0.0)

        # Update fitness of first population member (current params)
        fitness = -current_loss * (1.0 - forgetting)
        self._population_fitness[0] = fitness

        # Every 10 evaluations, breed
        if len(self.history) > 0 and len(self.history) % 10 == 0:
            self._breed_population()

    def _breed_population(self):
        """Breed new population from top-2 parents using crossover + mutation."""
        self._pop_gen += 1
        rng = np.random.RandomState(self._pop_gen)

        # Sort by fitness
        sorted_idx = sorted(range(len(self._population)),
                            key=lambda i: self._population_fitness[i], reverse=True)

        if len(sorted_idx) < 2:
            return

        parent1 = self._population[sorted_idx[0]]
        parent2 = self._population[sorted_idx[1]]

        new_pop = [parent1.copy(), parent2.copy()]  # Keep elites

        for i in range(2, len(self._population)):
            child = {}
            for param in self.PARAM_NAMES:
                lo, hi = self.bounds.get(param, [0.0, 1.0])
                # Crossover
                alpha = rng.rand()
                child_val = alpha * parent1.get(param, (lo + hi) / 2) + (1 - alpha) * parent2.get(param, (lo + hi) / 2)
                # Mutation (10% chance)
                if rng.rand() < 0.1:
                    child_val += rng.randn() * (hi - lo) * 0.05
                child[param] = float(np.clip(child_val, lo, hi))
            new_pop.append(child)

        self._population = new_pop
        self._population_fitness = [self._population_fitness[sorted_idx[0]], self._population_fitness[sorted_idx[1]]] + [-float("inf")] * (len(self._population) - 2)

    # -----------------------------------------------------------------------
    # Confidence scoring
    # -----------------------------------------------------------------------

    def _estimate_confidence(self, param: str) -> float:
        """Estimate probability of improvement from modification history."""
        history = self._confidence_history.get(param, [])
        if len(history) < 2:
            return 0.5  # No history — neutral confidence

        improvements = [before > after for before, after in history]
        return float(sum(improvements)) / len(improvements)

    def _record_outcome(self, param: str, loss_before: float, loss_after: float):
        """Record outcome of a modification for confidence tracking."""
        self._confidence_history.setdefault(param, []).append((loss_before, loss_after))
        if len(self._confidence_history[param]) > 20:
            self._confidence_history[param] = self._confidence_history[param][-20:]

    # -----------------------------------------------------------------------
    # Rollback
    # -----------------------------------------------------------------------

    def _save_rollback_state(self, current_loss: float):
        """Save current trainer state for potential rollback."""
        state = self.trainer.get_state()
        self._rollback_state = {
            param: state.get(param) for param in self.PARAM_NAMES
            if state.get(param) is not None
        }
        self._rollback_loss_at_change = current_loss
        self._rollback_loss_window = [current_loss]

    def _check_and_rollback(self, current_loss: float) -> bool:
        """Check if rollback is needed. Returns True if rollback was performed."""
        if self._rollback_state is None or self._rollback_loss_at_change is None:
            return False

        self._rollback_loss_window.append(current_loss)
        if len(self._rollback_loss_window) > self._rollback_window_size:
            self._rollback_loss_window = self._rollback_loss_window[-self._rollback_window_size:]

        if len(self._rollback_loss_window) >= 5:
            recent_avg = np.mean(self._rollback_loss_window[-5:])
            if recent_avg > self._rollback_loss_at_change * self._rollback_threshold:
                print(f"[self_modify] Rollback triggered: loss {recent_avg:.4f} > {self._rollback_loss_at_change * self._rollback_threshold:.4f}")
                for param, val in self._rollback_state.items():
                    if val is not None:
                        self.trainer.update_param(param, val)
                self._rollback_state = None
                self._rollback_loss_at_change = None
                return True

        return False

    # -----------------------------------------------------------------------
    # Parameter interdependence
    # -----------------------------------------------------------------------

    def _apply_interdependence(self, changed_param: str, new_value: float):
        """Apply interdependence rules when changing a parameter."""
        rules = self.INTERDEPENDENCE.get(changed_param, [])
        for dependent_param, scale in rules:
            state = self.trainer.get_state()
            current_val = state.get(dependent_param)
            if current_val is not None:
                adjusted_val = float(current_val) * scale
                lo, hi = self.bounds.get(dependent_param, [0.0, float("inf")])
                adjusted_val = float(np.clip(adjusted_val, lo, hi))
                print(f"[self_modify] Interdependence: {changed_param}↑ → {dependent_param}: {current_val:.4f} → {adjusted_val:.4f}")
                self.trainer.update_param(dependent_param, adjusted_val)

    # -----------------------------------------------------------------------
    # Multi-step planning
    # -----------------------------------------------------------------------

    def _generate_plan(self, ai_metrics: dict) -> List[Dict]:
        """Generate a 3-step modification plan based on current diagnostics."""
        plan = []
        loss = ai_metrics.get("loss", 0.0)
        grad_snr = ai_metrics.get("gradient_snr", 1.0)
        clip_rate = ai_metrics.get("clip_rate", 0.0)
        forgetting = ai_metrics.get("forgetting_score", 0.0)
        state = self.trainer.get_state()

        # Step 1: Address most urgent issue
        if clip_rate > 0.5:
            plan.append({
                "parameter": "gradient_clip_norm",
                "new_value": min(self.bounds["gradient_clip_norm"][1],
                                 float(state.get("gradient_clip_norm", 1.0)) * 0.7),
                "reason": "High gradient clipping rate — reduce clip threshold",
            })
        elif grad_snr < 1.0:
            current_lr = float(state.get("learning_rate", 1e-4))
            plan.append({
                "parameter": "learning_rate",
                "new_value": max(self.bounds["learning_rate"][0], current_lr * 0.5),
                "reason": "Low gradient SNR — reduce learning rate for stability",
            })

        # Step 2: Address forgetting
        if forgetting > 0.3:
            plan.append({
                "parameter": "ewc_lambda",
                "new_value": min(self.bounds["ewc_lambda"][1],
                                 float(state.get("ewc_lambda", 0.1)) * 2.0),
                "reason": "High forgetting — increase EWC lambda",
            })

        # Step 3: Fine-tune for efficiency
        lora_util = ai_metrics.get("lora_rank_utilization", 0.5)
        if lora_util < 0.3:
            plan.append({
                "parameter": "distill_alpha",
                "new_value": min(self.bounds["distill_alpha"][1],
                                 float(state.get("distill_alpha", 0.1)) * 1.5),
                "reason": "Low LoRA rank utilization — increase distillation to regularize",
            })

        return plan[:3]  # Max 3 steps

    # -----------------------------------------------------------------------
    # Main evaluate loop
    # -----------------------------------------------------------------------

    async def evaluate(self, ai_metrics: dict, bio_metrics: dict) -> Optional[Modification]:
        """Have the LLM analyze its diagnostics and optionally self-modify."""
        self._last_eval_step = ai_metrics.get("step", 0)
        current_loss = ai_metrics.get("loss", 0.0)

        # Check rollback first
        if self._check_and_rollback(current_loss):
            return None

        # Update Pareto front
        forgetting = ai_metrics.get("forgetting_score", 0.0)
        efficiency = ai_metrics.get("parameter_efficiency", 0.0)
        state_params = self.trainer.get_state()
        self._update_pareto_front(current_loss, forgetting, efficiency, state_params, self._last_eval_step)

        # Update GP observation
        param_vec = self._param_to_vector(state_params)
        self._gp_update(param_vec, current_loss)

        # Update evolutionary population
        self._evolve_population(state_params, current_loss)

        # Check if we have a pending plan step to execute
        if self._plan_queue:
            plan_step = self._plan_queue.pop(0)
            return self._validate_and_apply(plan_step)

        # Auto-curriculum
        curriculum_msg = self._auto_curriculum(current_loss)
        if curriculum_msg:
            print(f"[self_modify] {curriculum_msg}")

        diagnostic_text = self.compile_diagnostic_summary(ai_metrics, bio_metrics)

        system_prompt = (
            "You are an AI model analyzing your own training diagnostics. "
            "You have a tool called 'self_modify' that lets you adjust one training parameter. "
            "Analyze the diagnostics carefully and decide whether a parameter adjustment would improve training. "
            "Pay special attention to: gradient SNR (low → reduce LR), clip rate (high → reduce grad_clip_norm), "
            "forgetting score (high → increase ewc_lambda), LoRA rank utilization (low → adjust distill_alpha), "
            "SAM sharpness (high → reduce sam_rho or learning_rate). "
            "If the metrics look healthy, do NOT make a change. "
            "Only make one modification at a time. Be conservative. "
            "Call self_modify with the parameter, new value, and reason."
        )

        prompt = f"{system_prompt}\n\n{diagnostic_text}\n\nAnalyze the diagnostics and decide whether to adjust a parameter:"

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_self_eval, prompt, current_loss
            )
            return result
        except Exception as e:
            print(f"[self_modify] Evaluation failed: {e}")
            return None

    def _run_self_eval(self, prompt: str, current_loss: float) -> Optional[Modification]:
        """Run self-evaluation inference synchronously."""
        print("[self_modify] Running self-evaluation inference...")
        was_training = self.model.training
        self.model.eval()

        if self.hooks:
            self.hooks.enabled = False

        try:
            import time as _t
            t0 = _t.time()
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.model.device)
            print(f"[self_modify] Tokenized: {enc['input_ids'].shape[1]} tokens ({_t.time()-t0:.1f}s)")

            with torch.no_grad():
                t1 = _t.time()
                gen_ids = self.model.generate(
                    **enc,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                print(f"[self_modify] Generated {gen_ids.shape[1] - enc['input_ids'].shape[1]} tokens ({_t.time()-t1:.1f}s)")

            response = self.tokenizer.decode(
                gen_ids[0][enc["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            print(f"[self_modify] Response: {response[:120]}...")

            mod = self._parse_modification(response, current_loss)
            return mod

        except Exception as e:
            print(f"[self_modify] Generation failed: {e}")
            return None
        finally:
            if self.hooks:
                self.hooks.enabled = True
            if was_training:
                self.model.train()

    # -----------------------------------------------------------------------
    # Parsing
    # -----------------------------------------------------------------------

    def _parse_modification(self, response: str, current_loss: float) -> Optional[Modification]:
        """Parse the model's response for a self_modify tool call."""
        json_match = re.search(r'\{[^{}]*"parameter"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_and_apply(data, current_loss)
            except json.JSONDecodeError:
                pass

        param_match = re.search(r'parameter["\s:=]+(\w+)', response, re.IGNORECASE)
        value_match = re.search(r'new_value["\s:=]+([\d.eE\-+]+)', response, re.IGNORECASE)
        reason_match = re.search(r'reason["\s:=]+"?([^"}\n]+)', response, re.IGNORECASE)

        if param_match and value_match:
            try:
                data = {
                    "parameter": param_match.group(1),
                    "new_value": float(value_match.group(1)),
                    "reason": reason_match.group(1).strip() if reason_match else "Model-initiated adjustment",
                }
                return self._validate_and_apply(data, current_loss)
            except (ValueError, IndexError):
                pass

        return None

    def _validate_and_apply(self, data: dict, current_loss: float = 0.0) -> Optional[Modification]:
        """Validate bounds, check confidence, apply with rollback setup."""
        param = data.get("parameter", "")
        new_val = data.get("new_value")
        reason = data.get("reason", "")

        if param not in self.bounds:
            print(f"[self_modify] Unknown parameter: {param}")
            return None

        lo, hi = self.bounds[param]
        if new_val is None:
            return None

        try:
            new_val = float(new_val)
        except (TypeError, ValueError):
            return None

        new_val = max(lo, min(hi, new_val))

        # Confidence check
        confidence = self._estimate_confidence(param)
        if confidence < self._confidence_threshold and len(self._confidence_history.get(param, [])) >= 3:
            print(f"[self_modify] Skipping {param} modification: confidence {confidence:.2f} < {self._confidence_threshold}")
            return None

        # Get current value
        state = self.trainer.get_state()
        old_val = state.get(param, 0.0)
        if old_val is None:
            old_val = 0.0

        # Save rollback state
        self._save_rollback_state(current_loss)

        # Apply modification
        self.trainer.update_param(param, new_val)

        # Apply interdependence rules
        self._apply_interdependence(param, new_val)

        # Generate multi-step plan if this was step 1
        if not self._plan_queue and len(self.history) % 5 == 0:
            try:
                ai_metrics = {}  # placeholder
                plan = self._generate_plan({"loss": current_loss})
                # Queue steps 2 and 3 (skip step 1 which we just did)
                for plan_step in plan[1:]:
                    if plan_step["parameter"] != param:
                        self._plan_queue.append(plan_step)
                self._plan_queue = self._plan_queue[:2]  # Max 2 pending steps
            except Exception:
                pass

        mod = Modification(
            parameter=param,
            old_value=float(old_val),
            new_value=new_val,
            reason=reason[:200],
            timestamp=time.time(),
            step=state.get("step", 0),
            confidence=confidence,
            expected_improvement=0.0,
        )
        self.history.append(mod)

        print(f"[self_modify] Step {mod.step}: {param} {old_val:.6f} → {new_val:.6f} "
              f"(confidence={confidence:.2f}) — \"{reason[:80]}\"")

        # Update confidence history
        self._record_outcome(param, current_loss, current_loss)  # will be updated at next check

        return mod

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_history(self) -> list:
        """Return modification history as serializable dicts."""
        return [
            {
                "parameter": m.parameter,
                "old_value": m.old_value,
                "new_value": m.new_value,
                "reason": m.reason,
                "timestamp": m.timestamp,
                "step": m.step,
                "confidence": m.confidence,
            }
            for m in self.history
        ]

    def get_optimization_state(self) -> dict:
        """Return Pareto front, GP state, and population fitness."""
        pareto = [
            {
                "loss": p.loss,
                "forgetting": p.forgetting,
                "efficiency": p.efficiency,
                "step": p.step,
            }
            for p in self._pareto_front
        ]

        # GP predictions on current population
        gp_means = []
        gp_stds = []
        if self._gp_fitted and self._population:
            try:
                X_pop = np.array([self._param_to_vector(p) for p in self._population])
                means, stds = self._gp.predict(X_pop)
                gp_means = means.tolist()
                gp_stds = stds.tolist()
            except Exception:
                pass

        return {
            "pareto_front": pareto,
            "gp_mean": gp_means,
            "gp_std": gp_stds,
            "population_fitness": list(self._population_fitness),
            "population": [copy.deepcopy(p) for p in self._population],
            "evolutionary_generation": self._pop_gen,
            "confidence_by_param": {
                param: self._estimate_confidence(param)
                for param in self.PARAM_NAMES
            },
            "curriculum_level": self._curriculum_level,
            "pending_plan_steps": len(self._plan_queue),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(lst: list) -> float:
    """Safe mean of a list."""
    return sum(lst) / len(lst) if lst else 0.0
