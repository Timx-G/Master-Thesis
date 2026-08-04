"""
System model components for UAV-MEC NOMA offloading simulation.

Models:
  - UAVModel: UAV flight dynamics and energy consumption
  - NOMAChannel: Air-to-ground channel with NOMA uplink
  - TaskGenerator: Stochastic computation task generation
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def noise_power_dbm(
    bandwidth_hz: float,
    noise_psd_dbm_hz: float = -174.0,
    noise_figure_db: float = 7.0,
) -> float:
    """Return receiver noise power in dBm for the requested bandwidth."""
    if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be finite and positive")
    if not math.isfinite(noise_psd_dbm_hz):
        raise ValueError("noise_psd_dbm_hz must be finite")
    if not math.isfinite(noise_figure_db):
        raise ValueError("noise_figure_db must be finite")
    return noise_psd_dbm_hz + noise_figure_db + 10.0 * math.log10(bandwidth_hz)


def noise_power_w(
    bandwidth_hz: float,
    noise_psd_dbm_hz: float = -174.0,
    noise_figure_db: float = 7.0,
) -> float:
    """Return receiver noise power in watts for the requested bandwidth."""
    power_dbm = noise_power_dbm(
        bandwidth_hz,
        noise_psd_dbm_hz,
        noise_figure_db,
    )
    try:
        power_w = 10.0 ** ((power_dbm - 30.0) / 10.0)
    except OverflowError as exc:
        raise ValueError("noise power must be finite and positive") from exc
    if not math.isfinite(power_w) or power_w <= 0:
        raise ValueError("noise power must be finite and positive")
    return power_w


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UAVConfig:
    """UAV physical parameters."""
    height: float = 100.0           # fixed altitude (m)
    v_max: float = 20.0             # max horizontal speed (m/s)
    f_uav: float = 8e9              # MEC computing capacity (cycles/s); K=1 scenario uses full 8GHz
    # Note: multi-UAV experiments (K=2) use ScenarioConfig.f_per_uav=4e9/UAV (total=8GHz split)
    # Simplified flight power: P_fly = c1 + c2 * v^2
    c1: float = 80.0                # constant hovering power (W)
    c2: float = 0.12                # velocity-dependent coefficient (W/(m/s)^2)
    dt: float = 1.0                 # time-slot duration (s)


@dataclass
class ChannelConfig:
    """Air-to-ground NOMA channel parameters."""
    bandwidth: float = 5e6          # total bandwidth (Hz, 5 MHz LTE single carrier)
    environment: str = "urban"
    noise_psd_dbm_hz: float = -174.0  # thermal noise power spectral density (dBm/Hz)
    noise_figure_db: float = 7.0    # Receiver noise figure (dB); NF=7 dB is mid-range for UAV receivers
    # LoS probability: P_LoS = 1 / (1 + a * exp(-b * (theta - a)))
    a_los: float = 9.61             # environment param a (Urban, Al-Hourani 2014 Table I)
    b_los: float = 0.16             # environment param b
    # Path-loss
    fc: float = 2e9                 # carrier frequency (Hz)
    eta_los: float = 1.0            # additional LoS loss (dB)
    eta_nlos: float = 20.0          # additional NLoS loss (dB)
    # NOMA power
    p_max: float = 0.2              # max device transmit power (W, 23 dBm)
    pathloss_model: str = "al_hourani"

    def __post_init__(self) -> None:
        if not isinstance(self.environment, str) or not self.environment.strip():
            raise ValueError("environment must be a non-empty string")
        if self.pathloss_model not in {"al_hourani", "3gpp_uma_av_audited"}:
            raise ValueError(
                "pathloss_model must be al_hourani or 3gpp_uma_av_audited"
            )


@dataclass
class TaskConfig:
    """Computation task parameters."""
    arrival_prob: float = 0.8       # per-slot task arrival probability (dense IoT)
    arrival_mode: str = "bernoulli"
    period_choices: tuple[int, ...] = (1,)
    period_jitter_slots: int = 0
    data_min: float = 400e3         # min data size (bits) = 50 KB
    data_max: float = 2000e3        # max data size (bits) = 250 KB
    comp_min: float = 500.0         # min computation density (cycles/bit)
    comp_max: float = 1000.0        # max computation density (cycles/bit)
    t_min: float | None = None
    t_max: float = 0.8              # deadline (s) — Phase 0 calibrated (Greedy=70%, gap=8.9pp)
    f_local: float = 1e9            # local CPU frequency (cycles/s, 1 GHz)
    kappa: float = 1e-27            # energy coefficient for local computing
    result_ratio: float = 0.0       # no result return unless an independent downlink is configured

    def __post_init__(self) -> None:
        if self.arrival_mode not in {"bernoulli", "periodic"}:
            raise ValueError("arrival_mode must be bernoulli or periodic")
        try:
            self.period_choices = tuple(self.period_choices)
        except TypeError as exc:
            raise ValueError(
                "period_choices must contain positive integers"
            ) from exc
        if not self.period_choices or any(
            not isinstance(period, int)
            or isinstance(period, bool)
            or period <= 0
            for period in self.period_choices
        ):
            raise ValueError("period_choices must contain positive integers")
        if (
            not isinstance(self.period_jitter_slots, int)
            or isinstance(self.period_jitter_slots, bool)
            or self.period_jitter_slots < 0
        ):
            raise ValueError("period_jitter_slots must be a non-negative integer")
        if (
            not isinstance(self.t_max, (int, float))
            or isinstance(self.t_max, bool)
            or not math.isfinite(self.t_max)
            or self.t_max <= 0.0
        ):
            raise ValueError("t_max must be positive and finite")
        if self.t_min is not None and (
            not isinstance(self.t_min, (int, float))
            or isinstance(self.t_min, bool)
            or not math.isfinite(self.t_min)
            or self.t_min <= 0.0
            or self.t_min > self.t_max
        ):
            raise ValueError("t_min must be positive and no larger than t_max")
        if not math.isfinite(self.result_ratio) or not 0.0 <= self.result_ratio <= 1.0:
            raise ValueError("result_ratio must be finite in [0, 1]")


@dataclass
class ScenarioConfig:
    """Top-level scenario parameters."""
    num_users: int = 20
    area_size: float = 800.0        # square area side (m)  [V3: 800m for multi-UAV]
    num_slots: int = 100            # episode length T
    seed: Optional[int] = None
    # V2: clustered users
    num_clusters: int = 3           # number of user hotspots
    cluster_layout: str = "fixed"   # named layout; formal H2 uses pairwise_overlap
    cluster_std: float = 50.0       # std of Gaussian clusters (m)
    fixed_cluster_centers: bool = True  # use deterministic cluster layout for reproducibility
    # V2: battery budget
    battery_budget: float = 9000.0  # UAV battery budget (J): hover=8000J, tight for movement
    # V2: task queue
    max_queue_age: int = 3          # max slots a task can wait before timeout
    # V2: dynamic burst pattern
    dynamic_burst: bool = False     # enable rotating cluster bursts
    burst_cycle: int = 20           # slots per burst phase (legacy deterministic)
    burst_rate: float = 0.9         # task arrival rate in bursting cluster
    cool_rate: float = 0.2          # task arrival rate in cooling clusters
    # V3 (Phase 0B): non-stationary random burst
    random_burst: bool = False      # True = random period U[min,max] + random cluster
    burst_period_min: int = 8       # min slots between burst switches
    burst_period_max: int = 15      # max slots between burst switches
    # V3 (Phase 0B): user mobility
    user_mobility: bool = False     # enable user random walk
    user_speed: float = 2.0         # user movement speed (m/s)
    # Multi-UAV
    num_uavs: int = 2              # K=2 cooperative UAVs
    f_per_uav: float = 4e9         # MEC compute per UAV (total=K*f=8GHz)
    collision_dist: float = 30.0   # minimum inter-UAV distance (m)
    collision_penalty: float = 5.0 # reward penalty for collision
    # Grouped NOMA (Phase 0B+): equal-width subcarriers with 2-user NOMA pairs
    n_subcarriers: int = 3         # number of OFDMA subcarriers sharing total channel bandwidth
    users_per_sub: int = 2         # NOMA users per subcarrier (pair size)
    uav: UAVConfig = field(default_factory=UAVConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    def __post_init__(self) -> None:
        if isinstance(self.uav, dict):
            self.uav = UAVConfig(**self.uav)
        if isinstance(self.channel, dict):
            self.channel = ChannelConfig(**self.channel)
        if isinstance(self.task, dict):
            self.task = TaskConfig(**self.task)


# Fixed cluster layouts for reproducible experiments.
FIXED_CLUSTER_CENTERS_3 = np.array([
    [200.0, 200.0],   # Cluster 0: lower-left
    [600.0, 200.0],   # Cluster 1: lower-right
    [400.0, 600.0],   # Cluster 2: upper-center
])

FIXED_CLUSTER_CENTERS_4 = np.array([
    [200.0, 200.0],   # Cluster 0: lower-left
    [600.0, 200.0],   # Cluster 1: lower-right
    [200.0, 600.0],   # Cluster 2: upper-left
    [600.0, 600.0],   # Cluster 3: upper-right
])

# UAV hover positions per K (one per cluster, at cluster center)
FIXED_CLUSTER_CENTERS_5 = np.array([
    [160.0, 160.0],   # Cluster 0: lower-left
    [640.0, 160.0],   # Cluster 1: lower-right
    [160.0, 640.0],   # Cluster 2: upper-left
    [640.0, 640.0],   # Cluster 3: upper-right
    [400.0, 400.0],   # Cluster 4: center
])

UAV_HOVER_POSITIONS = {
    1: [(400.0, 400.0)],                         # center of 800x800 area
    2: [(267.0, 333.0), (533.0, 267.0)],       # optimized spread for 3 clusters
    3: [(200.0, 200.0), (600.0, 200.0), (400.0, 600.0)],  # one per cluster
    4: [(200.0, 200.0), (600.0, 200.0), (200.0, 600.0), (600.0, 600.0)],  # one per cluster (4 clusters)
    5: [(160.0, 160.0), (640.0, 160.0), (160.0, 640.0), (640.0, 640.0), (400.0, 400.0)],  # one per cluster (5 clusters)
}


# ---------------------------------------------------------------------------
# UAV flight model
# ---------------------------------------------------------------------------

class UAVModel:
    """UAV flight dynamics (fixed altitude, 2-D horizontal movement)."""

    def __init__(self, cfg: UAVConfig, area_size: float):
        self.cfg = cfg
        self.area_size = area_size
        # Initial position: center of the area at height H
        self.x: float = area_size / 2.0
        self.y: float = area_size / 2.0

    def reset(self):
        self.x = self.area_size / 2.0
        self.y = self.area_size / 2.0

    @property
    def position(self) -> np.ndarray:
        """3-D position [x, y, H]."""
        return np.array([self.x, self.y, self.cfg.height])

    def update_position(self, dx: float, dy: float) -> np.ndarray:
        """Move UAV by (dx, dy) within speed and area constraints.

        Args:
            dx, dy: desired displacement (m) in one time-slot.
        Returns:
            Actual displacement applied [dx', dy'].
        """
        max_dist = self.cfg.v_max * self.cfg.dt
        dist = math.hypot(dx, dy)
        if dist > max_dist:
            scale = max_dist / dist
            dx *= scale
            dy *= scale

        # Clip to area boundaries
        new_x = np.clip(self.x + dx, 0.0, self.area_size)
        new_y = np.clip(self.y + dy, 0.0, self.area_size)
        actual_dx = new_x - self.x
        actual_dy = new_y - self.y
        self.x = new_x
        self.y = new_y
        return np.array([actual_dx, actual_dy])

    def fly_energy(self, dx: float, dy: float) -> float:
        """Energy consumed for one time-slot movement (Joules).

        P_fly = c1 + c2 * v^2,  E = P_fly * dt
        """
        v = math.hypot(dx, dy) / self.cfg.dt
        p_fly = self.cfg.c1 + self.cfg.c2 * v ** 2
        return p_fly * self.cfg.dt


# ---------------------------------------------------------------------------
# NOMA air-to-ground channel
# ---------------------------------------------------------------------------

class NOMAChannel:
    """Air-to-ground channel with NOMA uplink (SIC decoding)."""

    def __init__(self, cfg: ChannelConfig, uav_height: float):
        if not math.isfinite(uav_height) or uav_height <= 0:
            raise ValueError("uav_height must be finite and positive")
        self.cfg = cfg
        self.H = uav_height
        self._validate_carrier_frequency()
        self.c = 3e8  # speed of light

    @property
    def noise_linear(self) -> float:
        """Full-band noise retained as a writable legacy interface."""
        return noise_power_w(
            self.cfg.bandwidth,
            self.cfg.noise_psd_dbm_hz,
            self.cfg.noise_figure_db,
        )

    @noise_linear.setter
    def noise_linear(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("noise_linear must be finite and positive")
        full_band_offset_db = noise_power_dbm(
            self.cfg.bandwidth,
            0.0,
            self.cfg.noise_figure_db,
        )
        full_band_dbm = 10.0 * math.log10(value) + 30.0
        self.cfg.noise_psd_dbm_hz = full_band_dbm - full_band_offset_db

    def noise_for_bandwidth(self, bandwidth_hz: float) -> float:
        """Return receiver noise power in watts for a channel bandwidth."""
        return noise_power_w(
            bandwidth_hz,
            self.cfg.noise_psd_dbm_hz,
            self.cfg.noise_figure_db,
        )

    def _validate_carrier_frequency(self) -> None:
        if not math.isfinite(self.cfg.fc) or self.cfg.fc <= 0:
            raise ValueError("fc must be finite and positive")

    # -- Path loss ----------------------------------------------------------

    def _los_probability(self, d_horizontal: float) -> float:
        """LoS probability given horizontal distance."""
        if d_horizontal < 1e-3:
            return 1.0
        theta_deg = math.degrees(math.atan2(self.H, d_horizontal))
        p_los = 1.0 / (1.0 + self.cfg.a_los *
                        math.exp(-self.cfg.b_los * (theta_deg - self.cfg.a_los)))
        return p_los

    def _uma_av_los_probability(self, d_horizontal: float) -> float:
        """TR 36.777 Annex B UMa-AV LOS probability for 22.5--100 m."""
        if not 22.5 < self.H <= 100.0:
            raise ValueError(
                "3gpp_uma_av_audited LOS probability requires 22.5 < height <= 100 m"
            )
        if d_horizontal <= 0.0:
            return 1.0
        log_height = math.log10(self.H)
        p1_m = 4300.0 * log_height - 3800.0
        d1_m = max(460.0 * log_height - 700.0, 18.0)
        if d_horizontal <= d1_m:
            return 1.0
        return d1_m / d_horizontal + math.exp(-d_horizontal / p1_m) * (
            1.0 - d1_m / d_horizontal
        )

    def _uma_av_mean_loss_db(self, d_horizontal: float, d_3d: float) -> float:
        """Probability-weighted TR 36.777 Annex B UMa-AV large-scale loss."""
        if not 10.0 < self.H <= 100.0:
            raise ValueError(
                "3gpp_uma_av_audited NLOS pathloss requires 10 < height <= 100 m"
            )
        if d_horizontal > 4_000.0:
            raise ValueError(
                "3gpp_uma_av_audited pathloss requires horizontal distance <= 4 km"
            )
        fc_ghz = self.cfg.fc / 1e9
        los_db = 28.0 + 22.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)
        nlos_db = (
            -17.5
            + (46.0 - 7.0 * math.log10(self.H)) * math.log10(d_3d)
            + 20.0 * math.log10(40.0 * math.pi * fc_ghz / 3.0)
        )
        p_los = self._uma_av_los_probability(d_horizontal)
        return p_los * los_db + (1.0 - p_los) * nlos_db

    def path_loss(self, d_horizontal: float) -> float:
        """Return channel power gain using dB-averaged LoS/NLoS loss."""
        if not math.isfinite(d_horizontal) or d_horizontal < 0:
            raise ValueError("d_horizontal must be finite and non-negative")
        self._validate_carrier_frequency()
        d_3d = math.hypot(d_horizontal, self.H)
        if self.cfg.pathloss_model == "3gpp_uma_av_audited":
            return 10.0 ** (-self._uma_av_mean_loss_db(d_horizontal, d_3d) / 10.0)
        if self.cfg.pathloss_model != "al_hourani":
            raise ValueError("unsupported pathloss_model")
        fspl_db = 20.0 * math.log10(
            4.0 * math.pi * self.cfg.fc * d_3d / self.c
        )
        p_los = self._los_probability(d_horizontal)
        mean_loss_db = (
            fspl_db
            + p_los * self.cfg.eta_los
            + (1.0 - p_los) * self.cfg.eta_nlos
        )
        return 10.0 ** (-mean_loss_db / 10.0)

    # -- NOMA rates ---------------------------------------------------------

    def compute_rates(
        self,
        uav_pos: np.ndarray,
        user_positions: np.ndarray,
        powers: np.ndarray,
        sic_eps: float = 0.0,
    ) -> np.ndarray:
        """Compute NOMA uplink rates with optional imperfect SIC.

        sic_eps: residual interference factor ε ∈ [0,1].
          0.0 = perfect SIC (default).
          With imperfect SIC, already-decoded users contribute
          ε × their received power as residual interference.
        """
        N = len(powers)
        h = np.zeros(N)
        for i in range(N):
            d_h = np.linalg.norm(uav_pos[:2] - user_positions[i])
            h[i] = self.path_loss(d_h)

        received = powers * h
        order = np.argsort(-received)   # descending

        rates = np.zeros(N)
        B = self.cfg.bandwidth

        for rank, idx in enumerate(order):
            # interference from weaker users (decoded after)
            interference = sum(received[order[m]] for m in range(rank + 1, N))
            # residual from stronger users (decoded before) — imperfect SIC
            residual = sic_eps * sum(received[order[m]] for m in range(rank))
            sinr = received[idx] / (interference + residual + self.noise_linear)
            sinr = max(sinr, 1e-10)
            rates[idx] = B * math.log2(1.0 + sinr)

        return rates

    def compute_ofdma_rates(
        self,
        uav_pos: np.ndarray,
        user_positions: np.ndarray,
        powers: np.ndarray,
    ) -> np.ndarray:
        """Compute OFDMA rates: each active user gets B/N_active bandwidth,
        no inter-user interference (fair OMA baseline).
        """
        N = len(powers)
        n_active = max(np.sum(powers > 1e-6), 1)
        bw_per_user = self.cfg.bandwidth / n_active
        noise_per_user = self.noise_for_bandwidth(bw_per_user)

        rates = np.zeros(N)
        for i in range(N):
            if powers[i] < 1e-6:
                continue
            d_h = np.linalg.norm(uav_pos[:2] - user_positions[i])
            h_i = self.path_loss(d_h)
            snr = powers[i] * h_i / noise_per_user
            snr = max(snr, 1e-10)
            rates[i] = bw_per_user * math.log2(1.0 + snr)

        return rates


    def compute_multi_uav_ofdma_rates(
        self,
        uav_positions: list,
        user_positions: np.ndarray,
        powers: np.ndarray,
        user_assignment: np.ndarray,
    ) -> np.ndarray:
        """OFDMA multi-UAV: each UAV gets B/K bandwidth, users within a UAV
        share their portion equally. No inter-user or inter-UAV interference."""
        N = len(powers)
        K = len(uav_positions)
        rates = np.zeros(N)
        bw_per_uav = self.cfg.bandwidth / K

        for k in range(K):
            users_k = [i for i in range(N)
                       if user_assignment[i] == k and powers[i] > 1e-6]
            if not users_k:
                continue
            bw_per_user = bw_per_uav / len(users_k)
            noise_per_user = self.noise_for_bandwidth(bw_per_user)
            for i in users_k:
                d_h = np.linalg.norm(
                    np.array(uav_positions[k][:2]) - user_positions[i])
                h_i = self.path_loss(d_h)
                snr = powers[i] * h_i / noise_per_user
                snr = max(snr, 1e-10)
                rates[i] = bw_per_user * math.log2(1.0 + snr)

        return rates

    def compute_grouped_noma_rates(
        self,
        uav_positions: list,        # [(x1,y1,H), (x2,y2,H)]
        user_positions: np.ndarray, # (N,2)
        user_assignment: np.ndarray, # (N,) which UAV serves user i
        active_mask: np.ndarray,    # (N,) bool: has active task
        n_sub: int = 5,             # number of OFDMA subcarriers
        users_per_sub: int = 2,     # users per NOMA pair
        sic_eps: float = 0.02,
    ) -> tuple:
        """Grouped NOMA: OFDMA subcarriers + 2-user NOMA pairs.

        Full sharing: both UAVs use all n_sub subcarriers.
        Each subcarrier carries one 2-user NOMA pair per UAV.
        Cross-UAV pairs on the same subcarrier cause inter-UAV interference.

        Returns:
            rates: np.ndarray (N,) — per-user rate in bps (0 for unserved)
            served: np.ndarray (N,) bool — whether user was selected for service
        """
        N = len(active_mask)
        K = len(uav_positions)
        p_max = self.cfg.p_max
        max_served_per_uav = n_sub * users_per_sub

        rates = np.zeros(N)
        served = np.zeros(N, dtype=bool)
        bw_sub = self.cfg.bandwidth / n_sub

        noise_sub = self.noise_for_bandwidth(bw_sub)

        # Precompute channel gains h[i][k] for all users to all UAVs
        h = np.zeros((N, K))
        for i in range(N):
            if not active_mask[i]:
                continue
            for k in range(K):
                d_h = np.linalg.norm(
                    np.array(uav_positions[k][:2]) - user_positions[i])
                h[i][k] = self.path_loss(d_h)

        # Per-UAV: select top users by channel gain, form pairs
        uav_served = {}   # k -> list of served user indices (ordered)
        uav_pairs = {}    # k -> list of (strong_user, weak_user) tuples

        for k in range(K):
            # Active users assigned to this UAV
            candidates = [i for i in range(N)
                          if user_assignment[i] == k and active_mask[i]]
            if not candidates:
                uav_served[k] = []
                uav_pairs[k] = []
                continue

            # Sort by channel gain to own UAV (descending) — best channels first
            candidates.sort(key=lambda i: h[i][k], reverse=True)
            # Select top users (up to max_served_per_uav)
            selected = candidates[:max_served_per_uav]
            uav_served[k] = selected

            for i in selected:
                served[i] = True

            # Form pairs: strongest with weakest (maximize SIC gain difference)
            # Pair formation: user[0] with user[-1], user[1] with user[-2], etc.
            n_sel = len(selected)
            pairs = []
            for p in range(n_sel // 2):
                i_strong = selected[p]            # high gain
                i_weak = selected[n_sel - 1 - p]  # low gain
                pairs.append((i_strong, i_weak))
            # If odd number of users, last one is solo
            if n_sel % 2 == 1:
                pairs.append((selected[n_sel // 2], None))
            uav_pairs[k] = pairs

        # Compute rates for each subcarrier
        for sub_idx in range(n_sub):
            # Collect pairs on this subcarrier (one per UAV)
            sub_pairs = {}  # k -> (strong, weak) or (solo, None)
            for k in range(K):
                if sub_idx < len(uav_pairs[k]):
                    sub_pairs[k] = uav_pairs[k][sub_idx]
                # else: this UAV has no pair on this subcarrier

            # Compute cross-UAV interference on this subcarrier
            # For UAV k: interference from all users of other UAVs on this sub
            for k in sub_pairs:
                pair_k = sub_pairs[k]
                i_strong, i_weak = pair_k

                # Cross-UAV interference at UAV_k from other UAVs' pairs
                cross_interf = 0.0
                for k2 in sub_pairs:
                    if k2 == k:
                        continue
                    i_s2, i_w2 = sub_pairs[k2]
                    if i_s2 is not None:
                        cross_interf += p_max * h[i_s2][k]
                    if i_w2 is not None:
                        cross_interf += p_max * h[i_w2][k]

                if i_weak is not None:
                    # 2-user NOMA pair: SIC decode strong first, then weak
                    recv_s = p_max * h[i_strong][k]
                    recv_w = p_max * h[i_weak][k]

                    # Strong user: interference from weak pair-mate + cross-UAV
                    sinr_s = recv_s / (recv_w + cross_interf + noise_sub)
                    rates[i_strong] = bw_sub * math.log2(1 + max(sinr_s, 1e-10))

                    # Weak user: strong already decoded (SIC), residual + cross-UAV
                    sinr_w = recv_w / (sic_eps * recv_s + cross_interf + noise_sub)
                    rates[i_weak] = bw_sub * math.log2(1 + max(sinr_w, 1e-10))
                else:
                    # Solo user on this subcarrier (OMA)
                    if i_strong is not None:
                        recv = p_max * h[i_strong][k]
                        sinr = recv / (cross_interf + noise_sub)
                        rates[i_strong] = bw_sub * math.log2(
                            1 + max(sinr, 1e-10))

        return rates, served

    def compute_multi_uav_rates(
        self,
        uav_positions: list,        # [(x1,y1,H), (x2,y2,H)]
        user_positions: np.ndarray, # (N,2)
        powers: np.ndarray,         # (N,) transmit powers
        user_assignment: np.ndarray, # (N,) which UAV serves user i (0 or 1)
        sic_eps: float = 0.02,
    ) -> np.ndarray:
        """Compute rates with inter-UAV interference (shared frequency band).

        User i served by UAV_k:
          SINR_i = h_{i,k}·p_i / (I_intra + I_inter + σ²)
          I_intra: NOMA intra-UAV interference (SIC-decoded, partially removed)
          I_inter: cross-UAV interference (CANNOT be removed by SIC)
        """
        N = len(powers)
        K = len(uav_positions)
        rates = np.zeros(N)
        B = self.cfg.bandwidth  # shared 5MHz

        # Precompute all channel gains: h[i][k] = gain from user i to UAV k
        h = np.zeros((N, K))
        for i in range(N):
            for k in range(K):
                d_h = np.linalg.norm(
                    np.array(uav_positions[k][:2]) - user_positions[i])
                h[i][k] = self.path_loss(d_h)

        # For each UAV, compute NOMA rates for its assigned users
        for k in range(K):
            users_k = [i for i in range(N) if user_assignment[i] == k
                       and powers[i] > 1e-6]
            if not users_k:
                continue

            # Received power at UAV_k from its own users
            recv_k = np.array([powers[i] * h[i][k] for i in users_k])
            # SIC order: descending received power
            order = np.argsort(-recv_k)

            # Cross-UAV interference at UAV_k from ALL users of other UAVs
            cross_interference = 0.0
            for k2 in range(K):
                if k2 == k:
                    continue
                for j in range(N):
                    if user_assignment[j] == k2 and powers[j] > 1e-6:
                        cross_interference += powers[j] * h[j][k]

            for rank, local_idx in enumerate(order):
                user_i = users_k[local_idx]
                # Intra-UAV interference (from users decoded after)
                intra = sum(recv_k[order[m]] for m in range(rank + 1, len(order)))
                # Residual from already-decoded (imperfect SIC)
                residual = sic_eps * sum(recv_k[order[m]] for m in range(rank))
                # Total interference
                total_interference = intra + residual + cross_interference
                sinr = recv_k[local_idx] / (total_interference + self.noise_linear)
                sinr = max(sinr, 1e-10)
                rates[user_i] = B * math.log2(1.0 + sinr)

        return rates


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------

class TaskGenerator:
    """Stochastic computation task generation per user per slot."""

    def __init__(self, cfg: TaskConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def generate(self, num_users: int) -> dict:
        """Generate tasks for one time-slot.

        Returns dict with arrays of length num_users:
            has_task: bool, whether user has a new task
            data_bits: float, data size D_k (bits)
            comp_cycles: float, total required cycles C_k
        """
        has_task = self.rng.random(num_users) < self.cfg.arrival_prob
        data_bits = self.rng.uniform(self.cfg.data_min, self.cfg.data_max, num_users)
        density = self.rng.uniform(self.cfg.comp_min, self.cfg.comp_max, num_users)
        comp_cycles = data_bits * density

        # Zero out for users with no task
        data_bits = data_bits * has_task
        comp_cycles = comp_cycles * has_task

        return {
            "has_task": has_task,
            "data_bits": data_bits,
            "comp_cycles": comp_cycles,
        }


# ---------------------------------------------------------------------------
# Utility: local and offload computation metrics
# ---------------------------------------------------------------------------

def local_delay(comp_cycles: float, f_local: float) -> float:
    """Local computation delay (s)."""
    if comp_cycles <= 0:
        return 0.0
    return comp_cycles / f_local


def local_energy(comp_cycles: float, f_local: float, kappa: float) -> float:
    """Local computation energy (J).  E = kappa * f^2 * C."""
    if comp_cycles <= 0:
        return 0.0
    return kappa * (f_local ** 2) * comp_cycles


def offload_delay(
    data_bits: float,
    comp_cycles: float,
    rate: float,
    f_alloc: float,
    max_delay: float = 10.0,
) -> float:
    """Return a legacy-v1 upload-plus-MEC delay for historical scripts.

    This array/scalar API predates task-level rho and independent downlink
    rates.  It is forbidden in the thesis-v2 formal pipeline.  The value is
    clamped to ``max_delay`` to preserve the historical behavior.
    """
    warnings.warn(
        "legacy-v1 delay semantics; forbidden in thesis-v2 formal pipeline",
        DeprecationWarning,
        stacklevel=2,
    )
    if data_bits <= 0 or comp_cycles <= 0:
        return 0.0
    if rate <= 0 or f_alloc <= 0:
        return max_delay
    delay = data_bits / rate + comp_cycles / f_alloc
    return min(delay, max_delay)


def offload_energy(data_bits: float, power: float, rate: float) -> float:
    """Transmission energy for offloading (J).  E = p * (D/R)."""
    if data_bits <= 0 or rate <= 0:
        return 0.0
    return power * (data_bits / rate)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("model.py self-test")
    print("=" * 60)

    cfg = ScenarioConfig(seed=42)
    rng = np.random.default_rng(cfg.seed)

    # -- UAV --
    uav = UAVModel(cfg.uav, cfg.area_size)
    print(f"UAV initial position: {uav.position}")
    disp = uav.update_position(10, 15)
    print(f"After move (10,15): pos={uav.position}, disp={disp}")
    e_fly = uav.fly_energy(disp[0], disp[1])
    print(f"Flight energy: {e_fly:.2f} J")

    # -- Channel --
    ch = NOMAChannel(cfg.channel, cfg.uav.height)
    user_pos = rng.uniform(0, cfg.area_size, (cfg.num_users, 2))
    powers = np.full(cfg.num_users, cfg.channel.p_max)
    rates = ch.compute_rates(uav.position, user_pos, powers)
    print(f"\nNOMA rates (Mbps): {rates / 1e6}")
    print(f"  min={rates.min()/1e6:.2f}, max={rates.max()/1e6:.2f}, "
          f"mean={rates.mean()/1e6:.2f}")

    # -- Tasks --
    tg = TaskGenerator(cfg.task, rng)
    tasks = tg.generate(cfg.num_users)
    active = tasks["has_task"].sum()
    print(f"\nTasks: {int(active)}/{cfg.num_users} users active")
    for i in range(cfg.num_users):
        if tasks["has_task"][i]:
            d = tasks["data_bits"][i]
            c = tasks["comp_cycles"][i]
            t_loc = local_delay(c, cfg.task.f_local)
            e_loc = local_energy(c, cfg.task.f_local, cfg.task.kappa)
            f_alloc = cfg.uav.f_uav / max(active, 1)
            t_off = offload_delay(d, c, rates[i], f_alloc)
            e_off = offload_energy(d, powers[i], rates[i])
            print(f"  User {i}: D={d/1e3:.0f}KB, C={c:.0f}cyc | "
                  f"local: {t_loc:.3f}s/{e_loc:.6f}J | "
                  f"offload: {t_off:.3f}s/{e_off:.6f}J")

    print("\nmodel.py: ALL TESTS PASSED")
