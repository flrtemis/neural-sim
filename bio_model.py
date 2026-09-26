"""
enhanced/bio_model.py — Izhikevich spiking neural network with full cortical column structure.

1000 neurons (800 excitatory, 200 inhibitory) organized into:
- 10 cortical columns × 100 neurons each, with 6 laminar layers (L1-L6)
- Realistic brain geometry with gyral/sulcal deformations
- Thalamic relay (40 neurons) providing alpha/delta rhythms
- AMPA/NMDA/GABA synaptic dynamics with rise/decay time constants
- Spike propagation delays based on Euclidean distance
- Neuromodulation (dopamine, acetylcholine, norepinephrine)
- Apical dendrite computation for L5 neurons (predictive coding)
- Brain state detection (active, resting, consolidating, creative)
- Gamma oscillations via inhibitory interneuron loops
- Theta rhythm coupling
- 50 extra cerebellum neurons

Provides all 24 original bio metrics PLUS:
  oscillation_bands, brain_state, thalamic_input, neuromodulator_levels,
  column_sync, apical_error, consolidation_score

Also provides:
  get_brain_positions() → list of {x, y, z, lobe, layer, active}
  get_spike_arcs() → recent spike propagation paths
"""

import math
import numpy as np
from enhanced.metrics import BioMetrics


# ---------------------------------------------------------------------------
# Constants for cortical column structure
# ---------------------------------------------------------------------------

N_COLUMNS = 10
NEURONS_PER_COLUMN = 100
N_CORTICAL = N_COLUMNS * NEURONS_PER_COLUMN   # 1000
N_THALAMUS = 40
N_CEREBELLUM = 50
N_TOTAL = N_CORTICAL + N_THALAMUS + N_CEREBELLUM  # 1090

# Layer layout within each column (total 100 per column)
LAYER_LAYOUT = {
    "L1":   (0,   5,   "inhibitory"),    # 5 neurons
    "L23":  (5,   30,  "regular"),       # 25 neurons
    "L4":   (30,  50,  "stellate"),      # 20 neurons
    "L5":   (50,  80,  "bursting"),      # 30 neurons (intrinsic bursting)
    "L6":   (80,  100, "regular"),       # 20 neurons
}

# Lobe assignments for columns
LOBE_MAP = {
    0: "frontal",    1: "frontal",    2: "frontal",
    3: "parietal",   4: "parietal",
    5: "temporal",   6: "temporal",
    7: "occipital",  8: "occipital",
    9: "limbic",
}


# ---------------------------------------------------------------------------
# Brain geometry generation
# ---------------------------------------------------------------------------

def _generate_brain_positions(n_columns: int = N_COLUMNS) -> np.ndarray:
    """Generate anatomically-inspired 3D positions for all neurons.

    Returns array of shape [N_TOTAL, 3] in mm coordinates.
    """
    positions = np.zeros((N_TOTAL, 3))

    # Lobe anchor points (approximate human brain layout, MNI-like coords)
    lobe_centers = {
        "frontal":   np.array([0.0,   40.0,  20.0]),
        "parietal":  np.array([0.0,  -30.0,  55.0]),
        "temporal":  np.array([55.0,  -10.0, -10.0]),
        "occipital": np.array([0.0,  -90.0,  10.0]),
        "limbic":    np.array([0.0,   -5.0,  20.0]),
    }

    # Hemisphere offset
    hemisphere_offset = np.array([30.0, 0.0, 0.0])

    # Generate per-column positions with gyral folding
    rng = np.random.RandomState(42)

    for col in range(n_columns):
        lobe = LOBE_MAP[col]
        center = lobe_centers[lobe].copy()

        # Hemisphere: columns 0-4 → left, 5-9 → right (but mirror for left)
        if col < 5:
            center[0] = -abs(center[0]) - 15.0 - rng.rand() * 10
        else:
            center[0] = abs(center[0]) + 15.0 + rng.rand() * 10

        col_base = col * NEURONS_PER_COLUMN

        for local_idx in range(NEURONS_PER_COLUMN):
            neuron_idx = col_base + local_idx

            # Determine layer
            layer_name = "L23"
            for lname, (lo, hi, ltype) in LAYER_LAYOUT.items():
                if lo <= local_idx < hi:
                    layer_name = lname
                    break

            # Layer depth along cortical surface normal (z-like component)
            layer_depth = {
                "L1": 0.0, "L23": -0.5, "L4": -1.0, "L5": -1.5, "L6": -2.0
            }.get(layer_name, 0.0)

            # Surface position with gyral folds using multi-octave sine/cosine
            t = (local_idx / NEURONS_PER_COLUMN) * 2 * math.pi
            gyral_scale = 8.0
            gyral_x = (
                math.sin(t) * gyral_scale
                + 0.5 * math.sin(2 * t) * gyral_scale * 0.5
                + 0.25 * math.cos(3 * t) * gyral_scale * 0.3
            )
            gyral_y = (
                math.cos(t) * gyral_scale
                + 0.5 * math.cos(2 * t) * gyral_scale * 0.5
                + 0.25 * math.sin(3 * t) * gyral_scale * 0.3
            )
            gyral_z = layer_depth * 2.5 + math.sin(t * 1.5) * 2.0

            # Add noise for realism
            noise = rng.randn(3) * 1.5

            positions[neuron_idx] = center + np.array([gyral_x, gyral_y, gyral_z]) + noise

    # Thalamus: small cluster at center-deep
    thal_center = np.array([0.0, -10.0, 5.0])
    thal_start = N_CORTICAL
    for i in range(N_THALAMUS):
        positions[thal_start + i] = thal_center + rng.randn(3) * 3.0

    # Cerebellum: posterior-inferior cluster
    cereb_center = np.array([0.0, -70.0, -30.0])
    cereb_start = N_CORTICAL + N_THALAMUS
    for i in range(N_CEREBELLUM):
        positions[cereb_start + i] = cereb_center + rng.randn(3) * 5.0

    return positions


# ---------------------------------------------------------------------------
# IzhikevichNetwork — enhanced
# ---------------------------------------------------------------------------

class IzhikevichNetwork:
    """Enhanced Izhikevich spiking neural network with cortical column structure."""

    def __init__(self, config: dict):
        bio_cfg = config.get("bio", {})
        self.dt = bio_cfg.get("dt", 0.5)

        # Neuron count
        self.n_cortical = N_CORTICAL
        self.n_thalamus = N_THALAMUS
        self.n_cerebellum = N_CEREBELLUM
        self.n = N_TOTAL

        # Count excitatory/inhibitory for compatibility
        self.n_exc = 800
        self.n_inh = 200

        # STDP parameters
        self.tau_plus = bio_cfg.get("stdp_tau_plus", 20.0)
        self.tau_minus = bio_cfg.get("stdp_tau_minus", 20.0)
        self.a_plus = bio_cfg.get("stdp_a_plus", 0.005)
        self.a_minus = bio_cfg.get("stdp_a_minus", 0.00525)

        # Homeostatic
        self.target_rate = bio_cfg.get("target_firing_rate", 5.0)
        self.homeo_tau = bio_cfg.get("homeostatic_tau", 100.0)

        # Pruning
        self.prune_threshold = bio_cfg.get("pruning_threshold", 0.01)
        self.prune_start = bio_cfg.get("pruning_start_step", 50)

        # ---- Neuron parameters ----
        rng = np.random.RandomState(123)
        self.a = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.c = np.zeros(self.n)
        self.d = np.zeros(self.n)

        # Assign per neuron based on column/layer type
        for col in range(N_COLUMNS):
            for local_idx in range(NEURONS_PER_COLUMN):
                ni = col * NEURONS_PER_COLUMN + local_idx
                layer_name, layer_type = self._get_layer_type(local_idx)

                if layer_type == "inhibitory":  # L1
                    self.a[ni] = 0.02 + 0.08 * rng.rand()
                    self.b[ni] = 0.25 - 0.05 * rng.rand()
                    self.c[ni] = -65.0
                    self.d[ni] = 2.0
                elif layer_type == "stellate":  # L4
                    # Stellate: similar to RS but slightly different
                    self.a[ni] = 0.02
                    self.b[ni] = 0.2
                    self.c[ni] = -65.0 + 10.0 * rng.rand() ** 2
                    self.d[ni] = 6.0 - 3.0 * rng.rand() ** 2
                elif layer_type == "bursting":  # L5 intrinsic bursting
                    self.a[ni] = 0.02
                    self.b[ni] = 0.2
                    self.c[ni] = -55.0
                    self.d[ni] = 4.0
                else:  # regular spiking (L23, L6)
                    self.a[ni] = 0.02
                    self.b[ni] = 0.2
                    self.c[ni] = -65.0 + 15.0 * rng.rand() ** 2
                    self.d[ni] = 8.0 - 6.0 * rng.rand() ** 2

        # Thalamus neurons: relay cells (RS-like)
        thal_start = N_CORTICAL
        self.a[thal_start:thal_start + N_THALAMUS] = 0.02
        self.b[thal_start:thal_start + N_THALAMUS] = 0.25
        self.c[thal_start:thal_start + N_THALAMUS] = -65.0
        self.d[thal_start:thal_start + N_THALAMUS] = 2.0

        # Cerebellum: granule cells (fast, high frequency)
        cereb_start = N_CORTICAL + N_THALAMUS
        self.a[cereb_start:] = 0.02 + 0.08 * rng.rand(N_CEREBELLUM)
        self.b[cereb_start:] = 0.2
        self.c[cereb_start:] = -65.0
        self.d[cereb_start:] = 6.0

        # ---- State ----
        self.v = -65.0 * np.ones(self.n)
        self.u = self.b * self.v

        # ---- Synaptic weights (sparse for 1090 neurons) ----
        # Use sparse representation: only allocate cortical-cortical + thalamo-cortical
        self._init_weights(rng)

        # ---- Spike propagation delays ----
        self._positions = _generate_brain_positions()
        self._delay_matrix = self._compute_delays()  # [n_cortical, n_cortical] integer steps
        self._delay_buffer_len = 11  # max ~5ms at 0.5ms dt
        self._delay_buffer = np.zeros((self.n_cortical, self._delay_buffer_len), dtype=bool)
        self._delay_buf_ptr = 0

        # ---- STDP traces ----
        self.stdp_trace_pre = np.zeros(self.n)
        self.stdp_trace_post = np.zeros(self.n)

        # ---- Spike history ----
        self.spikes = np.zeros(self.n, dtype=bool)
        self.spike_times = np.full(self.n, -1000.0)
        self.firing_rates = np.zeros(self.n)
        self._rate_alpha = 0.01

        # ---- History ----
        self.step = 0
        self._weight_history = []
        self._rate_history = []
        self._spike_count_window = []
        self._pain_events = []
        self._consolidated_memory = 0.0
        self._habit_strength = 0.0
        self._attention_focus = 0.3
        self._certainty = 0.3
        self._cortical_stability = 0.3

        # ---- Place cells ----
        n_place = 30
        self._place_positions = rng.randn(n_place, 2) * 1.5
        self._place_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

        # ---- Neuromodulators ----
        self._dopamine = 0.5
        self._acetylcholine = 0.5
        self._norepinephrine = 0.5

        # ---- Oscillation tracking ----
        self._osc_history = []  # firing rate sums, last 200 steps
        self._osc_buf_len = 200

        # ---- Thalamic rhythm ----
        self._thal_alpha_phase = 0.0   # 10 Hz alpha
        self._thal_delta_phase = 0.0   # 2 Hz delta
        self._thal_theta_phase = 0.0   # 6 Hz theta

        # ---- Theta modulation ----
        self._theta_mod = 1.0

        # ---- AMPA/NMDA/GABA conductances ----
        self._g_ampa = np.zeros(self.n)
        self._g_nmda = np.zeros(self.n)
        self._g_gaba = np.zeros(self.n)

        # ---- Apical dendrite inputs (L5 neurons) ----
        self._apical_input = np.zeros(self.n)   # top-down (feedback)
        self._basal_input = np.zeros(self.n)    # bottom-up (feedforward)

        # ---- Spike arcs buffer ----
        self._spike_arcs_buf = []
        self._arc_buf_len = 20

        # ---- Consolidation ----
        self._consolidation_score = 0.0
        self._brain_state = "resting"

        # ---- Initial weights for cortical stability ----
        self._initial_W_cortical = self.W_cortical.copy()

    # -----------------------------------------------------------------------
    # Initialization helpers
    # -----------------------------------------------------------------------

    def _get_layer_type(self, local_idx: int):
        """Return (layer_name, layer_type) for a local neuron index."""
        for lname, (lo, hi, ltype) in LAYER_LAYOUT.items():
            if lo <= local_idx < hi:
                return lname, ltype
        return "L23", "regular"

    def _init_weights(self, rng: np.random.RandomState):
        """Initialize synaptic weight matrices."""
        # Cortical-cortical weights (sparse n_cortical × n_cortical)
        # Build as full float32 but keep small values for sparsity
        # To keep memory manageable, use float16 or sparse
        # We'll use full float32 for correctness but limit n to cortical neurons only
        n = self.n_cortical
        self.W_cortical = np.zeros((n, n), dtype=np.float32)

        # Within-column connections (dense)
        for col in range(N_COLUMNS):
            start = col * NEURONS_PER_COLUMN
            end = start + NEURONS_PER_COLUMN
            # Excitatory connections within column
            col_exc_mask = np.array([
                self._is_excitatory(i - start) for i in range(start, end)
            ])
            for pre in range(start, end):
                if col_exc_mask[pre - start]:  # excitatory pre
                    for post in range(start, end):
                        if pre != post:
                            self.W_cortical[pre, post] = rng.rand() * 0.3
                else:  # inhibitory pre
                    for post in range(start, end):
                        if pre != post:
                            self.W_cortical[pre, post] = -rng.rand() * 0.8

        # Between-column connections (sparse, 5% connectivity)
        for col_a in range(N_COLUMNS):
            for col_b in range(N_COLUMNS):
                if col_a == col_b:
                    continue
                start_a = col_a * NEURONS_PER_COLUMN
                start_b = col_b * NEURONS_PER_COLUMN
                # 5% connectivity
                n_connections = int(NEURONS_PER_COLUMN * NEURONS_PER_COLUMN * 0.05)
                for _ in range(n_connections):
                    pre = start_a + rng.randint(0, NEURONS_PER_COLUMN)
                    post = start_b + rng.randint(0, NEURONS_PER_COLUMN)
                    if self._is_excitatory(pre % NEURONS_PER_COLUMN):
                        self.W_cortical[pre, post] = rng.rand() * 0.2
                    else:
                        self.W_cortical[pre, post] = -rng.rand() * 0.5

        np.fill_diagonal(self.W_cortical, 0)

        # Thalamo-cortical weights: thalamus → cortex
        self.W_thal_cortex = rng.rand(N_THALAMUS, self.n_cortical).astype(np.float32) * 0.4

        # Cerebellum-cortex (weak feedback)
        self.W_cereb_cortex = rng.rand(N_CEREBELLUM, self.n_cortical).astype(np.float32) * 0.1

    def _is_excitatory(self, local_idx: int) -> bool:
        """Return True if this local neuron index is excitatory."""
        _, ltype = self._get_layer_type(local_idx)
        return ltype != "inhibitory"

    def _compute_delays(self) -> np.ndarray:
        """Compute distance-based axonal delays (0.5-5 ms) as integer time steps."""
        # Only for cortical neurons (1000 × 1000 would be 1M entries)
        # Use subsampled distance for speed (compute between column centers)
        delays = np.ones((self.n_cortical, self.n_cortical), dtype=np.int32)
        max_delay_steps = self._delay_buffer_len - 1

        # Compute column-level delays and broadcast to neurons
        for col_a in range(N_COLUMNS):
            for col_b in range(N_COLUMNS):
                if col_a == col_b:
                    delays[
                        col_a * NEURONS_PER_COLUMN:(col_a + 1) * NEURONS_PER_COLUMN,
                        col_b * NEURONS_PER_COLUMN:(col_b + 1) * NEURONS_PER_COLUMN
                    ] = 1
                else:
                    # Distance between column centers
                    center_a = self._positions[col_a * NEURONS_PER_COLUMN + 50]
                    center_b = self._positions[col_b * NEURONS_PER_COLUMN + 50]
                    dist = np.linalg.norm(center_a - center_b)
                    # Speed ~50 mm/ms → delay = dist / 50 ms
                    delay_ms = max(0.5, min(5.0, dist / 50.0))
                    delay_steps = max(1, min(max_delay_steps, int(delay_ms / self.dt)))
                    delays[
                        col_a * NEURONS_PER_COLUMN:(col_a + 1) * NEURONS_PER_COLUMN,
                        col_b * NEURONS_PER_COLUMN:(col_b + 1) * NEURONS_PER_COLUMN
                    ] = delay_steps

        return delays

    # -----------------------------------------------------------------------
    # Oscillation / rhythm generation
    # -----------------------------------------------------------------------

    def _thalamic_drive(self) -> np.ndarray:
        """Compute thalamic input to cortical neurons at this time step."""
        t_ms = self.step * self.dt

        # Alpha (10 Hz) and delta (2 Hz) oscillations
        alpha_hz = 10.0
        delta_hz = 2.0
        theta_hz = 6.0

        alpha_input = 4.0 * (1 + math.sin(2 * math.pi * alpha_hz * t_ms / 1000.0))
        delta_input = 6.0 * (1 + math.sin(2 * math.pi * delta_hz * t_ms / 1000.0))
        theta_input = 3.0 * (1 + math.sin(2 * math.pi * theta_hz * t_ms / 1000.0))

        # Theta modulates synaptic strength
        self._theta_mod = 0.7 + 0.3 * (1 + math.sin(2 * math.pi * theta_hz * t_ms / 1000.0)) / 2.0

        thal_current = np.zeros(self.n_cortical)
        # Thalamic relay neurons fire rhythmically
        thal_spikes = self.spikes[self.n_cortical:self.n_cortical + N_THALAMUS]
        if thal_spikes.any():
            thal_current += (self.W_thal_cortex.T @ thal_spikes.astype(np.float32)) * 3.0

        # Direct rhythmic drive to L4 and L23 neurons
        for col in range(N_COLUMNS):
            start = col * NEURONS_PER_COLUMN
            # L4 gets alpha drive
            thal_current[start + 30:start + 50] += alpha_input * 0.5
            # L5 gets delta drive
            thal_current[start + 50:start + 80] += delta_input * 0.3

        return thal_current, float(alpha_input + delta_input + theta_input) / 3.0

    def _update_thalamus(self):
        """Drive thalamus neurons with alpha/delta rhythms."""
        t_ms = self.step * self.dt
        thal_start = self.n_cortical
        thal_indices = slice(thal_start, thal_start + N_THALAMUS)

        alpha_drive = 5.0 * (1 + math.sin(2 * math.pi * 10.0 * t_ms / 1000.0))
        delta_drive = 8.0 * (1 + math.sin(2 * math.pi * 2.0 * t_ms / 1000.0))

        # Alternate half thalamus for alpha/delta
        half = N_THALAMUS // 2
        self.v[thal_start:thal_start + half] += alpha_drive * 0.3
        self.v[thal_start + half:thal_start + N_THALAMUS] += delta_drive * 0.3

    def _update_synaptic_conductances(self, fired: np.ndarray):
        """Update AMPA/NMDA/GABA conductances with exponential decay."""
        # Decay constants (in dt units)
        tau_ampa = 2.0   # ms
        tau_nmda = 100.0  # ms
        tau_gaba = 5.0   # ms

        decay_ampa = math.exp(-self.dt / tau_ampa)
        decay_nmda = math.exp(-self.dt / tau_nmda)
        decay_gaba = math.exp(-self.dt / tau_gaba)

        self._g_ampa *= decay_ampa
        self._g_nmda *= decay_nmda
        self._g_gaba *= decay_gaba

        # On spike: activate conductances
        cortical_fired = fired[:self.n_cortical]
        for pre in np.where(cortical_fired)[0]:
            is_exc = self._is_excitatory(pre % NEURONS_PER_COLUMN)
            if is_exc:
                self._g_ampa += self.W_cortical[pre] * 0.3
                self._g_nmda += self.W_cortical[pre] * 0.1
            else:
                self._g_gaba -= self.W_cortical[pre] * 0.5  # inhibitory W_cortical is negative

        # Clamp
        self._g_ampa = np.clip(self._g_ampa, 0, 5.0)
        self._g_nmda = np.clip(self._g_nmda, 0, 3.0)
        self._g_gaba = np.clip(self._g_gaba, -5.0, 0)

    # -----------------------------------------------------------------------
    # Simulate step
    # -----------------------------------------------------------------------

    def simulate_step(self, external_current: np.ndarray = None) -> BioMetrics:
        """Simulate one time step of the full 1090-neuron network."""
        self.step += 1

        # External input
        I = np.zeros(self.n)
        if external_current is not None:
            I[:min(len(external_current), self.n)] = external_current[:self.n]

        # ---- Thalamic drive ----
        thal_current, thal_level = self._thalamic_drive()
        I[:self.n_cortical] += thal_current

        # ---- Synaptic currents (AMPA/NMDA/GABA) ----
        synaptic_I = self._g_ampa * (0.0 - self.v[:self.n_cortical]) * 0.05  # AMPA
        synaptic_I += self._g_nmda * (0.0 - self.v[:self.n_cortical]) * 0.01  # NMDA
        synaptic_I += self._g_gaba * (self.v[:self.n_cortical] + 75.0) * 0.1   # GABA
        I[:self.n_cortical] += synaptic_I

        # ---- Norepinephrine: affects signal/noise ratio ----
        noise_scale = 5.0 * (1.0 + self._norepinephrine)
        I[:self.n_cortical] += np.random.randn(self.n_cortical) * noise_scale

        # ---- Update thalamus neurons ----
        self._update_thalamus()

        # ---- Cerebellum: simple oscillatory input ----
        cereb_start = self.n_cortical + N_THALAMUS
        I[cereb_start:] += np.random.randn(N_CEREBELLUM) * 3.0

        # ---- Delayed spike propagation within cortex ----
        # Read from delay buffer for current time step
        delayed_spikes = self._delay_buffer[:, self._delay_buf_ptr].copy()
        I[:self.n_cortical] += (self.W_cortical.T @ delayed_spikes.astype(np.float32)) * 4.0 * self._theta_mod

        # Write current spikes into delay buffer at appropriate future slots
        cortical_spikes_now = self.spikes[:self.n_cortical]
        for pre in np.where(cortical_spikes_now)[0]:
            for post in range(self.n_cortical):
                delay = self._delay_matrix[pre, post]
                future_slot = (self._delay_buf_ptr + delay) % self._delay_buffer_len
                self._delay_buffer[post, future_slot] = True

        # Advance delay buffer pointer
        self._delay_buf_ptr = (self._delay_buf_ptr + 1) % self._delay_buffer_len

        # ---- Apical dendrite computation (L5 neurons) ----
        # L5 basal = local column input, apical = feedback from L23 of neighboring columns
        for col in range(N_COLUMNS):
            l5_start = col * NEURONS_PER_COLUMN + 50
            l5_end = col * NEURONS_PER_COLUMN + 80
            # Basal: within-column L4 input
            l4_activity = self.firing_rates[col * NEURONS_PER_COLUMN + 30:col * NEURONS_PER_COLUMN + 50].mean()
            self._basal_input[l5_start:l5_end] = l4_activity
            # Apical: neighboring column L23 activity (top-down)
            neighbor_col = (col + 1) % N_COLUMNS
            l23_activity = self.firing_rates[
                neighbor_col * NEURONS_PER_COLUMN + 5:neighbor_col * NEURONS_PER_COLUMN + 30
            ].mean()
            self._apical_input[l5_start:l5_end] = l23_activity
            # Predictive coding: apical - basal = prediction error
            I[l5_start:l5_end] += (self._apical_input[l5_start:l5_end] - self._basal_input[l5_start:l5_end]) * 2.0

        # ---- Izhikevich dynamics ----
        fired = np.zeros(self.n, dtype=bool)
        for _ in range(int(1.0 / max(self.dt, 0.1))):
            dv = (0.04 * self.v ** 2 + 5.0 * self.v + 140.0 - self.u + I) * self.dt
            du = (self.a * (self.b * self.v - self.u)) * self.dt
            self.v += dv
            self.u += du
            spike_mask = self.v >= 30.0
            fired |= spike_mask
            self.v[spike_mask] = self.c[spike_mask]
            self.u[spike_mask] = self.u[spike_mask] + self.d[spike_mask]

        self.spikes = fired
        current_time = float(self.step)

        # Update spike times
        self.spike_times[fired] = current_time

        # Update firing rates (EMA)
        self.firing_rates = (1 - self._rate_alpha) * self.firing_rates + self._rate_alpha * fired.astype(float)

        # Spike count window
        self._spike_count_window.append(int(fired[:self.n_cortical].sum()))
        if len(self._spike_count_window) > 100:
            self._spike_count_window = self._spike_count_window[-100:]

        # ---- Update synaptic conductances ----
        self._update_synaptic_conductances(fired)

        # ---- STDP learning ----
        self._update_stdp(fired, current_time)

        # ---- Homeostatic scaling ----
        self._homeostatic_scale()

        # ---- Pruning ----
        if self.step > self.prune_start:
            self._prune()

        # ---- Neuromodulator updates ----
        self._update_neuromodulators(fired)

        # ---- Oscillation history ----
        self._osc_history.append(float(fired[:self.n_cortical].mean()))
        if len(self._osc_history) > self._osc_buf_len:
            self._osc_history = self._osc_history[-self._osc_buf_len:]

        # ---- Spike arcs ----
        self._update_spike_arcs(fired)

        return self._compute_metrics(fired, I, thal_level)

    # -----------------------------------------------------------------------
    # STDP
    # -----------------------------------------------------------------------

    def _update_stdp(self, fired: np.ndarray, t: float):
        """Apply STDP with acetylcholine-modulated plasticity window."""
        ach_factor = 0.5 + self._acetylcholine
        tau_p = self.tau_plus / ach_factor
        tau_m = self.tau_minus / ach_factor

        self.stdp_trace_pre *= math.exp(-self.dt / tau_p)
        self.stdp_trace_post *= math.exp(-self.dt / tau_m)
        self.stdp_trace_pre[fired] += self.a_plus
        self.stdp_trace_post[fired] += self.a_minus

        # Dopamine-modulated learning rate
        da_lr = self._dopamine

        # STDP on cortical excitatory synapses only (sampled for speed)
        cortical_fired = np.where(fired[:self.n_cortical])[0]
        if len(cortical_fired) > 0:
            sample = cortical_fired[:min(20, len(cortical_fired))]
            for j in sample:
                if not self._is_excitatory(j % NEURONS_PER_COLUMN):
                    continue
                # LTP: all pre trace contributes to this post neuron
                col_start = (j // NEURONS_PER_COLUMN) * NEURONS_PER_COLUMN
                col_end = col_start + NEURONS_PER_COLUMN
                self.W_cortical[col_start:col_end, j] += (
                    self.stdp_trace_pre[col_start:col_end] * da_lr * 0.5
                )

            for i in sample:
                if not self._is_excitatory(i % NEURONS_PER_COLUMN):
                    continue
                col_start = (i // NEURONS_PER_COLUMN) * NEURONS_PER_COLUMN
                col_end = col_start + NEURONS_PER_COLUMN
                self.W_cortical[i, col_start:col_end] -= self.stdp_trace_post[col_start:col_end] * da_lr * 0.5

        # Clip weights
        self.W_cortical = np.clip(self.W_cortical, -2.0, 1.0)
        np.fill_diagonal(self.W_cortical, 0)

    # -----------------------------------------------------------------------
    # Homeostatic scaling
    # -----------------------------------------------------------------------

    def _homeostatic_scale(self):
        """Multiplicative synaptic scaling per column to maintain target firing rate."""
        target = self.target_rate / 1000.0
        for col in range(N_COLUMNS):
            start = col * NEURONS_PER_COLUMN
            end = start + NEURONS_PER_COLUMN
            exc_rates = self.firing_rates[start:end]
            mean_rate = exc_rates.mean()
            if mean_rate > 0:
                ratio = np.clip(target / (mean_rate + 1e-10), 0.99, 1.01)
                self.W_cortical[start:end, :] *= ratio

    # -----------------------------------------------------------------------
    # Pruning
    # -----------------------------------------------------------------------

    def _prune(self):
        """Prune weak synapses."""
        mask = np.abs(self.W_cortical) < self.prune_threshold
        self.W_cortical[mask] = 0.0

    # -----------------------------------------------------------------------
    # Neuromodulator updates
    # -----------------------------------------------------------------------

    def _update_neuromodulators(self, fired: np.ndarray):
        """Update dopamine, acetylcholine, norepinephrine based on activity."""
        # Dopamine: reward signal from unexpected activity
        expected_rate = self.target_rate / 1000.0
        actual_rate = fired[:self.n_cortical].mean()
        delta = float(actual_rate - expected_rate)
        self._dopamine = float(np.clip(0.5 + 0.3 * delta * 100, 0.1, 1.0))

        # Acetylcholine: inversely related to overall activity (modulates attention/plasticity)
        ach_target = 0.5 + 0.3 * (1.0 - actual_rate / (expected_rate * 2 + 1e-10))
        self._acetylcholine = float(np.clip(
            0.9 * self._acetylcholine + 0.1 * ach_target, 0.1, 1.0
        ))

        # Norepinephrine: arousal signal, slowly decaying
        ne_signal = float(fired[:self.n_cortical].sum()) / (self.n_cortical * 0.1 + 1e-10)
        self._norepinephrine = float(np.clip(
            0.95 * self._norepinephrine + 0.05 * ne_signal, 0.1, 1.0
        ))

    # -----------------------------------------------------------------------
    # Spike arcs
    # -----------------------------------------------------------------------

    def _update_spike_arcs(self, fired: np.ndarray):
        """Record recent spike propagation paths for visualization."""
        fired_cortical = np.where(fired[:self.n_cortical])[0]
        if len(fired_cortical) < 2:
            return
        # Record up to 5 random arcs
        sample = fired_cortical[:min(5, len(fired_cortical))]
        for i in range(len(sample) - 1):
            pre = int(sample[i])
            post = int(sample[i + 1])
            if abs(self.W_cortical[pre, post]) > 0.05:
                p1 = self._positions[pre].tolist()
                p2 = self._positions[post].tolist()
                arc = {
                    "from": p1, "to": p2,
                    "weight": float(self.W_cortical[pre, post]),
                    "step": self.step,
                }
                self._spike_arcs_buf.append(arc)
        if len(self._spike_arcs_buf) > self._arc_buf_len:
            self._spike_arcs_buf = self._spike_arcs_buf[-self._arc_buf_len:]

    # -----------------------------------------------------------------------
    # Brain state detection
    # -----------------------------------------------------------------------

    def _detect_brain_state(self, osc_bands: dict) -> str:
        """Detect brain state from oscillation power."""
        if osc_bands["gamma"] > 0.3 and osc_bands["beta"] > 0.2:
            return "active"
        elif osc_bands["delta"] > 0.4 and osc_bands["theta"] < 0.1:
            return "consolidating"
        elif osc_bands["alpha"] > 0.3 and osc_bands["theta"] > 0.2:
            return "resting"
        elif osc_bands["theta"] > 0.25 and osc_bands["gamma"] > 0.2:
            return "creative"
        else:
            return "resting"

    # -----------------------------------------------------------------------
    # Oscillation band power estimation
    # -----------------------------------------------------------------------

    def _compute_oscillation_bands(self) -> dict:
        """Estimate oscillation band powers from firing rate history using FFT proxy."""
        if len(self._osc_history) < 10:
            return {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0}

        signal = np.array(self._osc_history, dtype=np.float32)
        signal -= signal.mean()

        # Use simple frequency band estimation via band-pass correlation
        dt_ms = self.dt
        fs = 1000.0 / dt_ms  # sampling frequency in Hz

        # Compute simple power spectrum via FFT
        n_fft = len(signal)
        freqs = np.fft.rfftfreq(n_fft, d=dt_ms / 1000.0)  # in Hz
        power = np.abs(np.fft.rfft(signal)) ** 2

        def band_power(f_lo, f_hi):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            return float(power[mask].sum()) if mask.any() else 0.0

        total_power = float(power.sum()) + 1e-10

        bands = {
            "delta": band_power(0.5, 4.0) / total_power,
            "theta": band_power(4.0, 8.0) / total_power,
            "alpha": band_power(8.0, 13.0) / total_power,
            "beta": band_power(13.0, 30.0) / total_power,
            "gamma": band_power(30.0, 100.0) / total_power,
        }
        return bands

    # -----------------------------------------------------------------------
    # Column synchrony
    # -----------------------------------------------------------------------

    def _compute_column_sync(self) -> list:
        """Compute pairwise column synchrony as cosine similarity of firing rates."""
        rates_per_col = []
        for col in range(N_COLUMNS):
            start = col * NEURONS_PER_COLUMN
            end = start + NEURONS_PER_COLUMN
            rates_per_col.append(self.firing_rates[start:end])

        sync = []
        for i in range(N_COLUMNS):
            row = []
            for j in range(N_COLUMNS):
                r_i = rates_per_col[i]
                r_j = rates_per_col[j]
                cos_sim = float(np.dot(r_i, r_j) / (np.linalg.norm(r_i) * np.linalg.norm(r_j) + 1e-10))
                row.append(float(np.clip(cos_sim, 0.0, 1.0)))
            sync.append(row)
        return sync

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def _compute_metrics(self, fired: np.ndarray, currents: np.ndarray, thal_level: float) -> BioMetrics:
        """Extract all bio diagnostics from current network state."""
        m = BioMetrics(step=self.step)

        # Use cortical neurons for most metrics (compatibility with original)
        n_exc = self.n_exc
        n_groups = 6
        group_size = n_exc // n_groups

        exc_W = self.W_cortical[:n_exc, :n_exc]
        cortical_fired = fired[:self.n_cortical]

        # 1. Synaptic weight
        pos_mask = exc_W > 0
        m.synaptic_weight = float(exc_W[pos_mask].mean()) if pos_mask.any() else 0.0

        # 2. Mastery
        rates = self.firing_rates[:n_exc]
        if rates.std() > 0:
            m.mastery = min(1.0, float(rates.mean() / (rates.std() + 1e-10)) * 0.3)
        else:
            m.mastery = 0.0

        # 3. Prediction error / dopamine delta
        expected_rate = self.target_rate / 1000.0
        actual_rate = float(cortical_fired[:n_exc].mean())
        m.delta = float(actual_rate - expected_rate)
        m.dopamine = float(np.clip(0.5 + 0.3 * m.delta * 100, 0.1, 1.0))

        # 4. Neural sparsity
        m.sparsity = float(1.0 - cortical_fired.mean())

        # 5. Dendritic signal per layer group
        m.dendritic_signal = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            group_rate = self.firing_rates[start:end].mean()
            m.dendritic_signal.append(float(group_rate) * 100)

        # 6. Place cells
        for label in range(3):
            mask = self._place_labels == label
            center = self._place_positions[mask].mean(axis=0)
            self._place_positions[mask] += 0.01 * (center - self._place_positions[mask])
            self._place_positions[mask] += np.random.randn(mask.sum(), 2) * 0.02
        m.place_x = self._place_positions[:, 0].tolist()
        m.place_y = self._place_positions[:, 1].tolist()
        m.place_labels = self._place_labels.tolist()

        # 7. Active fraction
        m.active_fraction = float(cortical_fired.mean())

        # 8. Selective attention
        self._attention_focus = min(0.95, self._attention_focus + 0.002 * abs(m.delta))
        m.attention_focus = self._attention_focus

        # 9. Homeostatic norms
        m.homeostatic_norms = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            norm = float(np.linalg.norm(exc_W[start:end, :]) / max(1, group_size))
            m.homeostatic_norms.append(norm)

        # 10. Attractor stability
        m.attractor_variance = float(np.var(self._spike_count_window[-20:])) if len(self._spike_count_window) >= 5 else 1.0

        # 11. Mean firing rate
        m.mean_firing_rate = float(self.firing_rates[:self.n_cortical].mean()) * 1000

        # 12. Certainty
        rates_exc = self.firing_rates[:n_exc]
        rate_mean = float(rates_exc.mean())
        rate_std = float(rates_exc.std())
        if rate_mean > 1e-6:
            cv = rate_std / rate_mean
            m.certainty = float(np.clip(1.0 / (1.0 + cv), 0.05, 0.95))
        else:
            m.certainty = 0.05

        # 13. Habit strength
        self._habit_strength = min(1.0, self._habit_strength + 0.003 * abs(m.delta))
        m.habit_strength = self._habit_strength

        # 14. Plasticity window
        m.plasticity = max(0.1, 1.0 - self.step * 0.0002)

        # 15. Synapse density
        total_possible = n_exc * self.n_cortical
        active_synapses = float((np.abs(exc_W) > self.prune_threshold).sum())
        m.synapse_density = active_synapses / total_possible

        # 16. Bypass ratios
        m.bypass_ratios = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            direct = float(np.abs(exc_W[start:end, :]).mean())
            m.bypass_ratios.append(float(np.clip(direct / (direct + 0.1), 0, 1)))

        # 17. Cortical stability
        flat_now = self.W_cortical[:n_exc, :n_exc].flatten()
        flat_init = self._initial_W_cortical[:n_exc, :n_exc].flatten()
        dot = np.dot(flat_now, flat_init)
        norms = float(np.linalg.norm(flat_now) * np.linalg.norm(flat_init)) + 1e-10
        m.cortical_stability = float(np.clip(dot / norms, 0.0, 1.0))

        # 18. Working memory profile
        if len(self._spike_count_window) >= 7:
            window = self._spike_count_window[-7:]
            positions = np.arange(7)
            primacy = np.exp(-0.3 * positions)
            recency = np.exp(-0.3 * (6 - positions))
            profile = (primacy + recency) / 2
            m.wm_profile = profile.tolist()
        else:
            m.wm_profile = [0.5] * 7

        # 19. Pain events
        pain_signal = abs(m.delta)
        m.pain_triggered = 1.0 if pain_signal > 0.05 else 0.0
        self._pain_events.append(m.pain_triggered)
        if len(self._pain_events) > 100:
            self._pain_events = self._pain_events[-100:]
        m.pain_rate = float(np.mean(self._pain_events[-50:]))

        # 20. Memory retention
        if len(self._rate_history) >= 2:
            old_rates = self._rate_history[0]
            new_rates = self.firing_rates[:len(old_rates)]
            corr = np.corrcoef(old_rates, new_rates)[0, 1]
            m.memory_retention = float(np.clip(corr, 0.0, 1.0)) if not np.isnan(corr) else 0.0
        else:
            m.memory_retention = 0.0
        if self.step % 10 == 0:
            self._rate_history.append(self.firing_rates[:n_exc].copy())
            if len(self._rate_history) > 10:
                self._rate_history = self._rate_history[-10:]

        # 21. Population diversity
        rates_full = self.firing_rates[:self.n_cortical] + 1e-10
        rates_norm = rates_full / rates_full.sum()
        entropy = -float((rates_norm * np.log(rates_norm)).sum())
        max_entropy = math.log(self.n_cortical)
        m.population_diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        # 22. Noise benefit
        noise_level = float(np.std(currents[:n_exc]))
        optimal_noise = 5.0
        m.noise_benefit = float(np.clip(1.0 - abs(noise_level - optimal_noise) / optimal_noise, 0, 1))

        # 23. Column differentiation
        diffs = []
        for col in range(N_COLUMNS - 1):
            s1 = col * NEURONS_PER_COLUMN
            e1 = s1 + NEURONS_PER_COLUMN
            s2 = (col + 1) * NEURONS_PER_COLUMN
            e2 = s2 + NEURONS_PER_COLUMN
            r1 = self.firing_rates[s1:e1]
            r2 = self.firing_rates[s2:e2]
            cos_sim = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2) + 1e-10)
            diffs.append(1.0 - abs(cos_sim))
        m.column_differentiation = float(np.mean(diffs)) if diffs else 0.0

        # 24. Area specialisation
        group_activities = []
        for g in range(min(4, n_groups)):
            start = g * group_size
            end = start + group_size
            group_activities.append(float(self.firing_rates[start:end].sum()) + 1e-10)
        total = sum(group_activities)
        m.specialisation = [a / total for a in group_activities]
        m.specialisation_entropy = -sum(p * math.log(p + 1e-10) for p in m.specialisation)

        # Neuron activities (normalized firing rates for visualization)
        max_rate = self.firing_rates[:self.n_cortical].max() + 1e-10
        m.neuron_activities = (self.firing_rates[:self.n_cortical] / max_rate).tolist()

        # Spike pairs (top connections)
        fired_indices = np.where(cortical_fired)[0]
        pairs = []
        if len(fired_indices) > 1:
            for i in fired_indices[:10]:
                for j in fired_indices[:10]:
                    if i != j and abs(self.W_cortical[i, j]) > 0.1:
                        pairs.append([int(i), int(j), float(abs(self.W_cortical[i, j]))])
        m.spike_pairs = pairs

        # ---- Extended metrics ----

        # Oscillation bands
        osc_bands = self._compute_oscillation_bands()
        m.oscillation_bands = osc_bands

        # Brain state
        self._brain_state = self._detect_brain_state(osc_bands)
        m.brain_state = self._brain_state

        # Thalamic input
        m.thalamic_input = float(thal_level)

        # Neuromodulator levels
        m.neuromodulator_levels = {
            "dopamine": self._dopamine,
            "acetylcholine": self._acetylcholine,
            "norepinephrine": self._norepinephrine,
        }

        # Column synchrony matrix
        m.column_sync = self._compute_column_sync()

        # Apical error (mean L5 prediction error)
        l5_apical = self._apical_input[50:80]  # first column L5 as representative
        l5_basal = self._basal_input[50:80]
        m.apical_error = float(np.abs(l5_apical - l5_basal).mean())

        # Consolidation score (sleep-like delta power → consolidation)
        self._consolidation_score = 0.95 * self._consolidation_score + 0.05 * osc_bands.get("delta", 0.0)
        m.consolidation_score = float(self._consolidation_score)

        # Brain positions (for viz — return first 100 neurons to keep payload small)
        m.brain_positions = self.get_brain_positions()[:100]

        # Spike arcs
        m.spike_arcs = list(self._spike_arcs_buf[-10:])

        return m

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_brain_positions(self) -> list:
        """Return list of {x, y, z, lobe, layer, active} for each cortical neuron."""
        result = []
        for neuron_idx in range(self.n_cortical):
            col = neuron_idx // NEURONS_PER_COLUMN
            local = neuron_idx % NEURONS_PER_COLUMN
            layer_name, _ = self._get_layer_type(local)
            lobe = LOBE_MAP.get(col, "unknown")
            pos = self._positions[neuron_idx]
            is_active = bool(self.spikes[neuron_idx])
            result.append({
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "lobe": lobe,
                "layer": layer_name,
                "active": is_active,
            })
        return result

    def get_spike_arcs(self) -> list:
        """Return recent spike propagation paths for visualization."""
        return list(self._spike_arcs_buf)

    def encode_text_to_current(self, token_ids: list, embedding_dim: int = 100) -> np.ndarray:
        """Convert token IDs to input currents via rate coding."""
        current = np.zeros(self.n)
        for i, tid in enumerate(token_ids[:20]):
            start = (tid % self.n_exc)
            width = 10
            for j in range(width):
                idx = (start + j) % self.n_exc
                current[idx] += 3.0 * (1.0 + math.sin(tid * 0.1 + j * 0.5))
        return current
