"""
UAV-MEC NOMA Offloading Gymnasium Environment.

Observation (continuous):
    [uav_x, uav_y]                        (2)     UAV horizontal position
    [dist_1 .. dist_N]                     (N)     horizontal distance to each user
    [has_task_1 .. has_task_N]             (N)     current task indicator
    [data_1 .. data_N]                     (N)     normalised task data size
    [comp_1 .. comp_N]                     (N)     normalised task compute load
    [channel_gain_1 .. channel_gain_N]     (N)     normalised channel gain
    [time_remaining]                       (1)     fraction of episode left
    Total dim: 2 + 5*N + 1

Action (continuous, Box):
    [uav_dx, uav_dy]                       (2)     UAV movement (scaled by v_max*dt)
    [offload_ratio_1 .. offload_ratio_N]   (N)     0=local, 1=full offload
    [power_frac_1 .. power_frac_N]         (N)     fraction of p_max
    Total dim: 2 + 2*N

Reward (Plan A — CMDP):
    - penalty for deadline violations + UAV flight energy
    - Delay and user energy constraints handled by Lagrangian (in SAC agent)
"""

from __future__ import annotations

import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .model import (
    ChannelConfig,
    NOMAChannel,
    ScenarioConfig,
    TaskConfig,
    TaskGenerator,
    UAVConfig,
    UAVModel,
    local_delay,
    local_energy,
    offload_delay,
    offload_energy,
)


class UAVMECEnv(gym.Env):
    """UAV-MEC NOMA Offloading Environment.

    The agent controls a single UAV equipped with a MEC server.
    At each time-slot it decides:
      (1) where to fly,
      (2) per-user offloading ratio,
      (3) per-user NOMA transmit power.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cfg: Optional[ScenarioConfig] = None,
        render_mode: Optional[str] = None,
        w_task_success: float = 1.0,
        w_fly_energy: float = 0.01,
        use_ofdma: bool = False,       # if True: OFDMA rates (no NOMA interference)
        sic_eps: float = 0.02,         # imperfect SIC residual factor ε (default 2%)
    ):
        super().__init__()
        self.cfg = cfg or ScenarioConfig()
        self.render_mode = render_mode
        self.w_task_success = w_task_success
        self.w_fly_energy = w_fly_energy
        self.use_ofdma = use_ofdma
        self.sic_eps = sic_eps

        N = self.cfg.num_users

        # -- Action space: all in [-1, 1], rescaled internally --
        # [dx, dy, offload_1..N, power_frac_1..N]
        self.action_dim = 2 + 2 * N
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # -- Observation space (V2) --
        # [uav_x, uav_y, {dist, has_task, data, comp, channel, queue_age}_N,
        #  time_frac, battery_frac]
        self.obs_dim = 2 + 6 * N + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # -- Submodels (created in reset) --
        self.uav: Optional[UAVModel] = None
        self.channel: Optional[NOMAChannel] = None
        self.task_gen: Optional[TaskGenerator] = None
        self.rng: Optional[np.random.Generator] = None

        # -- State variables --
        self.user_positions: Optional[np.ndarray] = None
        self.current_tasks: Optional[dict] = None
        self.slot: int = 0

        # V2: task queue (per-user, max 1 pending task)
        self.task_queue_data: Optional[np.ndarray] = None    # (N,) data_bits
        self.task_queue_comp: Optional[np.ndarray] = None    # (N,) comp_cycles
        self.task_queue_age: Optional[np.ndarray] = None     # (N,) slots waiting
        self.task_queue_active: Optional[np.ndarray] = None  # (N,) bool

        # V2: battery
        self.battery_remaining: float = 0.0

        # -- Episode metrics --
        self.episode_total_delay: float = 0.0
        self.episode_total_energy: float = 0.0
        self.episode_tasks_served: int = 0
        self.episode_deadline_violations: int = 0
        self.episode_tasks_dropped: int = 0   # V2: queue timeout drops

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Determine RNG seed
        _seed = seed if seed is not None else self.cfg.seed
        self.rng = np.random.default_rng(_seed)

        # Create submodels
        self.uav = UAVModel(self.cfg.uav, self.cfg.area_size)
        self.uav.reset()
        self.channel = NOMAChannel(self.cfg.channel, self.cfg.uav.height)
        self.task_gen = TaskGenerator(self.cfg.task, self.rng)

        # V2: Clustered user positions (mixture of Gaussians)
        N = self.cfg.num_users
        nc = self.cfg.num_clusters
        cluster_centers = self.rng.uniform(
            self.cfg.cluster_std * 2,
            self.cfg.area_size - self.cfg.cluster_std * 2,
            (nc, 2)
        )
        assignments = self.rng.integers(0, nc, N)
        self.user_cluster_ids = assignments  # V2: store for dynamic burst
        self.cluster_centers = cluster_centers
        self.user_positions = np.zeros((N, 2))
        for i in range(N):
            c = cluster_centers[assignments[i]]
            pos = self.rng.normal(c, self.cfg.cluster_std)
            self.user_positions[i] = np.clip(pos, 0, self.cfg.area_size)

        # Generate first tasks
        self.current_tasks = self.task_gen.generate(N)
        self.slot = 0

        # V2: Init task queue
        self.task_queue_data = np.zeros(N)
        self.task_queue_comp = np.zeros(N)
        self.task_queue_age = np.zeros(N, dtype=int)
        self.task_queue_active = np.zeros(N, dtype=bool)
        # Seed queue with first arriving tasks
        for i in range(N):
            if self.current_tasks["has_task"][i]:
                self.task_queue_data[i] = self.current_tasks["data_bits"][i]
                self.task_queue_comp[i] = self.current_tasks["comp_cycles"][i]
                self.task_queue_age[i] = 0
                self.task_queue_active[i] = True

        # V2: Init battery
        self.battery_remaining = self.cfg.battery_budget

        # Reset metrics
        self.episode_total_delay = 0.0
        self.episode_total_energy = 0.0
        self.episode_tasks_served = 0
        self.episode_deadline_violations = 0
        self.episode_tasks_dropped = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.uav is not None, "Call reset() first"
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        N = self.cfg.num_users

        # ------ Parse action ------
        dx_norm, dy_norm = action[0], action[1]
        offload_ratios = (action[2 : 2 + N] + 1.0) / 2.0       # [-1,1] -> [0,1]
        power_fracs = (action[2 + N : 2 + 2 * N] + 1.0) / 2.0  # [-1,1] -> [0,1]

        # Scale UAV movement
        max_disp = self.cfg.uav.v_max * self.cfg.uav.dt
        dx = float(dx_norm * max_disp)
        dy = float(dy_norm * max_disp)

        # Move UAV
        actual_disp = self.uav.update_position(dx, dy)
        e_fly = self.uav.fly_energy(actual_disp[0], actual_disp[1])

        # Scale transmit powers
        powers = power_fracs * self.cfg.channel.p_max  # (N,)

        # ------ Compute uplink rates ------
        if self.use_ofdma:
            rates = self.channel.compute_ofdma_rates(
                self.uav.position, self.user_positions, powers)
        elif self.sic_eps > 0:
            rates = self.channel.compute_rates(
                self.uav.position, self.user_positions, powers,
                sic_eps=self.sic_eps)
        else:
            rates = self.channel.compute_rates(
                self.uav.position, self.user_positions, powers)

        # ------ Per-user computation (V2: queue-aware) ------
        total_delay = 0.0
        total_energy = 0.0
        total_sigmoid_cost = 0.0
        total_user_energy = 0.0
        user_qualities = []   # V2: per-user service quality for Jain fairness
        deadline_violations = 0
        tasks_served = 0
        tasks_dropped = 0

        # Process queued tasks: compute offloading/local for active users
        offloading_users = []
        for i in range(N):
            if self.task_queue_active[i] and offload_ratios[i] > 0.01:
                offloading_users.append(i)

        total_offload_cycles = 0.0
        off_cycles = np.zeros(N)
        for i in offloading_users:
            off_cycles[i] = offload_ratios[i] * self.task_queue_comp[i]
            total_offload_cycles += off_cycles[i]

        for i in range(N):
            if not self.task_queue_active[i]:
                continue

            tasks_served += 1
            alpha = float(offload_ratios[i])
            D_k = self.task_queue_data[i]
            C_k = self.task_queue_comp[i]
            f_local = self.cfg.task.f_local
            kappa = self.cfg.task.kappa

            C_local = (1.0 - alpha) * C_k
            t_local = local_delay(C_local, f_local)
            e_local = local_energy(C_local, f_local, kappa)

            D_off = alpha * D_k
            C_off = alpha * C_k
            if C_off > 0 and total_offload_cycles > 0:
                f_alloc = self.cfg.uav.f_uav * (C_off / total_offload_cycles)
                f_alloc = max(f_alloc, 1e3)
            else:
                f_alloc = self.cfg.uav.f_uav

            t_off = offload_delay(D_off, C_off, rates[i], f_alloc)
            e_off = offload_energy(D_off, float(powers[i]), rates[i])

            user_delay = max(t_local, t_off)
            user_energy = e_local + e_off

            total_delay += user_delay
            total_energy += user_energy
            total_user_energy += user_energy

            _k = 5.0
            _x = _k * (user_delay - self.cfg.task.t_max)
            sig_cost = 1.0 / (1.0 + np.exp(-np.clip(_x, -20, 20)))
            total_sigmoid_cost += sig_cost
            user_qualities.append(1.0 - sig_cost)  # V2: quality = 1 - sigmoid_cost

            if user_delay <= self.cfg.task.t_max:
                # Task completed within deadline: remove from queue
                self.task_queue_active[i] = False
                self.task_queue_age[i] = 0
            else:
                deadline_violations += 1
                self.task_queue_age[i] += 1
                # V2: drop if exceeded max queue age
                if self.task_queue_age[i] >= self.cfg.max_queue_age:
                    self.task_queue_active[i] = False
                    self.task_queue_age[i] = 0
                    tasks_dropped += 1

        # ------ V2: Jain Fairness Index ------
        if len(user_qualities) >= 2:
            q = np.array(user_qualities)
            jain_index = (q.sum()**2) / (len(q) * (q**2).sum() + 1e-10)
        else:
            jain_index = 1.0
        fairness_target = 0.7
        fairness_cost = max(0.0, fairness_target - jain_index)

        # ------ Reward (CMDP: throughput - flight cost) ------
        D_max = self.cfg.task.data_max
        E_max = 128.0
        throughput = 0.0
        for i in range(N):
            if self.task_queue_active[i] or self.task_queue_age[i] > 0:
                throughput += offload_ratios[i] * self.task_queue_data[i] / D_max
        reward = (
            self.w_task_success * throughput
            - self.w_fly_energy * e_fly / E_max
        )

        # ------ V2: Battery tracking ------
        self.battery_remaining -= e_fly
        battery_frac = max(self.battery_remaining / self.cfg.battery_budget, 0.0)
        # Battery depletion penalty: if battery < 10%, add penalty
        if battery_frac < 0.1:
            reward -= 2.0  # harsh penalty for near-depletion

        # ------ Update episode metrics ------
        self.episode_total_delay += total_delay
        self.episode_total_energy += total_energy + e_fly
        self.episode_tasks_served += tasks_served
        self.episode_deadline_violations += deadline_violations
        self.episode_tasks_dropped += tasks_dropped

        # ------ Advance time ------
        self.slot += 1
        terminated = self.slot >= self.cfg.num_slots
        truncated = False

        # V2: Generate new tasks with optional dynamic burst pattern
        if not terminated:
            new_tasks = self.task_gen.generate(N)
            # Dynamic burst: override per-user arrival based on cluster phase
            if self.cfg.dynamic_burst and hasattr(self, 'user_cluster_ids'):
                nc = self.cfg.num_clusters
                burst_idx = (self.slot // self.cfg.burst_cycle) % nc
                for i in range(N):
                    cid = self.user_cluster_ids[i]
                    if cid == burst_idx:
                        prob = self.cfg.burst_rate
                    else:
                        prob = self.cfg.cool_rate
                    new_tasks["has_task"][i] = self.rng.random() < prob
                    if not new_tasks["has_task"][i]:
                        new_tasks["data_bits"][i] = 0
                        new_tasks["comp_cycles"][i] = 0
            for i in range(N):
                if new_tasks["has_task"][i] and not self.task_queue_active[i]:
                    self.task_queue_data[i] = new_tasks["data_bits"][i]
                    self.task_queue_comp[i] = new_tasks["comp_cycles"][i]
                    self.task_queue_age[i] = 0
                    self.task_queue_active[i] = True
            self.current_tasks = new_tasks

        obs = self._get_obs()
        info = self._get_info()
        safe_tasks = max(tasks_served, 1)
        info.update({
            "slot_delay": total_delay,
            "slot_energy": total_energy,
            "slot_fly_energy": e_fly,
            "slot_tasks": tasks_served,
            "slot_violations": deadline_violations,
            "slot_delay_cost": total_sigmoid_cost / safe_tasks,
            "slot_energy_cost": total_user_energy / safe_tasks,
            "slot_fly_cost": e_fly / E_max,
            # V2 signals
            "slot_dropped": tasks_dropped,
            "battery_frac": battery_frac,
            "queue_active": int(self.task_queue_active.sum()),
            "jain_index": jain_index,
            "fairness_cost": fairness_cost,
        })

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        N = self.cfg.num_users
        uav_xy = np.array([self.uav.x, self.uav.y]) / self.cfg.area_size  # [0,1]

        # Distances (normalised by area diagonal)
        diag = math.sqrt(2) * self.cfg.area_size
        dists = np.zeros(N)
        for i in range(N):
            dists[i] = np.linalg.norm(self.uav.position[:2] - self.user_positions[i])
        dists_norm = dists / diag

        # V2: Task info from queue (not current_tasks)
        has_task = self.task_queue_active.astype(np.float32)
        data_norm = self.task_queue_data / self.cfg.task.data_max
        comp_norm = self.task_queue_comp / (
            self.cfg.task.data_max * self.cfg.task.comp_max
        )
        # V2: Queue age normalized by max_queue_age
        queue_age_norm = self.task_queue_age.astype(np.float32) / max(self.cfg.max_queue_age, 1)

        # Channel gains (log-scale, normalised)
        ch_gains = np.zeros(N)
        for i in range(N):
            d_h = np.linalg.norm(self.uav.position[:2] - self.user_positions[i])
            ch_gains[i] = self.channel.path_loss(d_h)
        # Log-normalise: map to roughly [0, 1]
        ch_log = np.log10(ch_gains + 1e-20)
        ch_min, ch_max = ch_log.min(), ch_log.max()
        if ch_max - ch_min > 1e-8:
            ch_norm = (ch_log - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = np.zeros(N)

        # Time fraction remaining + V2: battery fraction
        time_frac = np.array([1.0 - self.slot / self.cfg.num_slots])
        batt_frac = np.array([max(self.battery_remaining / self.cfg.battery_budget, 0.0)])

        obs = np.concatenate([
            uav_xy, dists_norm, has_task, data_norm, comp_norm, ch_norm,
            queue_age_norm,  # V2: per-user queue age
            time_frac, batt_frac,  # V2: battery
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        return {
            "slot": self.slot,
            "uav_pos": self.uav.position.copy(),
            "episode_total_delay": self.episode_total_delay,
            "episode_total_energy": self.episode_total_energy,
            "episode_tasks_served": self.episode_tasks_served,
            "episode_deadline_violations": self.episode_deadline_violations,
        }

    # ------------------------------------------------------------------
    # Render (text mode)
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            print(
                f"[Slot {self.slot:3d}/{self.cfg.num_slots}] "
                f"UAV=({self.uav.x:.1f}, {self.uav.y:.1f}) | "
                f"served={self.episode_tasks_served} "
                f"violations={self.episode_deadline_violations}"
            )


# ---------------------------------------------------------------------------
# Quick self-test: random agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("env.py self-test: random agent episode")
    print("=" * 60)

    cfg = ScenarioConfig(num_users=10, num_slots=100, seed=42)
    env = UAVMECEnv(cfg=cfg, render_mode="human")

    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}  (expected {env.obs_dim})")
    print(f"Action shape: {env.action_space.shape}  (expected ({env.action_dim},))")

    total_reward = 0.0
    for step in range(cfg.num_slots):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            env.render()

        if terminated or truncated:
            break

    env.render()
    print(f"\nEpisode done:")
    print(f"  Total reward:     {total_reward:.2f}")
    print(f"  Tasks served:     {info['episode_tasks_served']}")
    print(f"  Deadline viols:   {info['episode_deadline_violations']}")
    print(f"  Total delay (s):  {info['episode_total_delay']:.2f}")
    print(f"  Total energy (J): {info['episode_total_energy']:.2f}")

    # Verify obs/action space consistency
    obs2, _ = env.reset(seed=99)
    assert obs2.shape == (env.obs_dim,), f"Obs shape mismatch: {obs2.shape}"
    assert env.observation_space.contains(obs2), "Obs not in observation_space"

    # Run a second episode to ensure reset works
    for _ in range(10):
        a = env.action_space.sample()
        o, r, t, tr, i = env.step(a)
        assert env.observation_space.contains(o)

    print("\nenv.py: ALL TESTS PASSED")
