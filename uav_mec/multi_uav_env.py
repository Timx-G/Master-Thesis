"""
Multi-UAV MEC Environment with shared frequency and inter-UAV interference.

Supports K={2,3,4} UAVs, grouped NOMA, fixed hover positions.
K=4 uses 4-cluster layout; K=2,3 use 3-cluster layout.
"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .model import (ScenarioConfig, UAVModel, NOMAChannel, TaskGenerator,
                    local_delay, local_energy, offload_delay, offload_energy,
                    FIXED_CLUSTER_CENTERS_3, FIXED_CLUSTER_CENTERS_4, FIXED_CLUSTER_CENTERS_5,
                    UAV_HOVER_POSITIONS)


class MultiUAVMECEnv(gym.Env):
    """Multi-UAV MEC with shared frequency band and inter-UAV interference.

    Supports arbitrary K UAVs. UAVs are fixed (no trajectory).
    Action: per-UAV [cluster_w0, ..., cluster_w_{nc-2}, alpha_base]
    """

    def __init__(self, cfg: Optional[ScenarioConfig] = None):
        super().__init__()
        self.cfg = cfg or ScenarioConfig()
        N = self.cfg.num_users
        K = self.cfg.num_uavs
        nc = self.cfg.num_clusters

        # Per-agent action: [cluster_weights(nc-1), alpha_base] = nc dim
        self.agent_action_dim = nc  # nc-1 cluster weights + 1 alpha_base
        self.action_dim = K * self.agent_action_dim
        self.action_space = spaces.Box(-1, 1, (self.action_dim,), dtype=np.float32)

        # Obs per agent: [own_xy(2), mean_other_xy(2), cluster_loads(nc), battery(1), time(1)]
        self.agent_obs_dim = 2 + 2 + nc + 1 + 1  # = nc + 6
        self.obs_dim = K * self.agent_obs_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)

        self.uavs = None
        self.channel = None
        self.task_gen = None
        self.rng = None
        self.user_positions = None
        self.user_cluster_ids = None
        self.cluster_centers = None
        self.task_queue_active = None
        self.task_queue_data = None
        self.task_queue_comp = None
        self.task_queue_age = None
        self.slot = 0
        self.batteries = None
        self.ep_tasks = 0
        self.ep_violations = 0
        self.ep_delay = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _seed = seed if seed is not None else 42
        self.rng = np.random.default_rng(_seed)
        # Separate RNG for strategy-level randomness (e.g., random pairing).
        # Must be independent from self.rng so that random-baseline episodes
        # experience the same mobility / task-arrival sequence as all other baselines.
        self.strategy_rng = np.random.default_rng(_seed + 999_983)

        N = self.cfg.num_users
        K = self.cfg.num_uavs
        nc = self.cfg.num_clusters

        # UAVs: fixed hover positions from config
        self.uavs = []
        hover_pos = UAV_HOVER_POSITIONS.get(K)
        if hover_pos is None or len(hover_pos) < K:
            # Fallback: spread evenly
            hover_pos = [(self.cfg.area_size * (0.2 + 0.6 * k / max(K-1, 1)),
                          self.cfg.area_size * 0.5) for k in range(K)]
        for k in range(K):
            uav = UAVModel(self.cfg.uav, self.cfg.area_size)
            uav.x, uav.y = hover_pos[k]
            self.uavs.append(uav)

        self.channel = NOMAChannel(self.cfg.channel, self.cfg.uav.height)
        self.task_gen = TaskGenerator(self.cfg.task, self.rng)

        # Cluster layout: 4 clusters for K=4, 3 clusters otherwise
        if self.cfg.fixed_cluster_centers:
            if nc == 5:
                centers = FIXED_CLUSTER_CENTERS_5.copy()
            elif nc == 4:
                centers = FIXED_CLUSTER_CENTERS_4.copy()
            elif nc == 3:
                centers = FIXED_CLUSTER_CENTERS_3.copy()
            else:
                centers = self.rng.uniform(
                    self.cfg.cluster_std * 2,
                    self.cfg.area_size - self.cfg.cluster_std * 2, (nc, 2))
        else:
            centers = self.rng.uniform(
                self.cfg.cluster_std * 2,
                self.cfg.area_size - self.cfg.cluster_std * 2, (nc, 2))

        # Balanced user assignment to clusters
        assignments = np.array([i % nc for i in range(N)])
        self.rng.shuffle(assignments)
        self.user_cluster_ids = assignments
        self.cluster_centers = centers
        self.user_positions = np.zeros((N, 2))
        for i in range(N):
            pos = self.rng.normal(centers[assignments[i]], self.cfg.cluster_std)
            self.user_positions[i] = np.clip(pos, 0, self.cfg.area_size)

        # Task queue
        self.task_queue_active = np.zeros(N, dtype=bool)
        self.task_queue_data = np.zeros(N)
        self.task_queue_comp = np.zeros(N)
        self.task_queue_age = np.zeros(N, dtype=int)
        first_tasks = self.task_gen.generate(N)
        for i in range(N):
            if first_tasks["has_task"][i]:
                self.task_queue_active[i] = True
                self.task_queue_data[i] = first_tasks["data_bits"][i]
                self.task_queue_comp[i] = first_tasks["comp_cycles"][i]

        self.batteries = np.full(K, self.cfg.battery_budget)
        self.slot = 0
        self.ep_tasks = 0
        self.ep_violations = 0
        self.ep_delay = 0.0

        # Non-stationary burst state
        if self.cfg.random_burst:
            self.burst_cluster = self.rng.integers(0, nc)
            self.next_burst_switch = self.rng.integers(
                self.cfg.burst_period_min, self.cfg.burst_period_max + 1)
        else:
            self.burst_cluster = 0
            self.next_burst_switch = 0

        return self._get_obs(), {}

    def _assign_users(self):
        """Assign each user to nearest UAV."""
        N = self.cfg.num_users
        K = self.cfg.num_uavs
        assignment = np.zeros(N, dtype=int)
        for i in range(N):
            dists = np.array([np.linalg.norm(
                np.array([self.uavs[k].x, self.uavs[k].y]) - self.user_positions[i])
                for k in range(K)])
            min_d = dists.min()
            candidates = np.where(np.abs(dists - min_d) < 1.0)[0]
            assignment[i] = self.rng.choice(candidates)
        return assignment

    def _compute_powers(self, user_assignment):
        """All active users transmit at p_max."""
        N = self.cfg.num_users
        powers = np.zeros(N)
        for i in range(N):
            if self.task_queue_active[i]:
                powers[i] = self.cfg.channel.p_max
        return powers

    def _compute_alphas(self, agent_actions, user_assignment, rates, served):
        """Delay-equalization crossing-point offloading ratio using actual grouped NOMA rates."""
        N = self.cfg.num_users
        K = self.cfg.num_uavs
        f_local = self.cfg.task.f_local
        alphas = np.zeros(N)

        for k in range(K):
            act = agent_actions[k]
            # Alpha base is last element of per-agent action
            alpha_base = 0.05 + 0.9 * (act[-1] + 1) / 2

            served_k = [i for i in range(N)
                        if user_assignment[i] == k and served[i]
                        and rates[i] > 1.0
                        and self.task_queue_data[i] > 1e-3
                        and self.task_queue_comp[i] > 1e-3]
            if not served_k:
                continue

            n_offloading = len(served_k)
            f_alloc = self.cfg.f_per_uav / max(n_offloading, 1)
            f_alloc = max(f_alloc, 1e3)

            for i in served_k:
                D_i = self.task_queue_data[i]
                C_i = self.task_queue_comp[i]
                R_i = rates[i]
                t_local_full = C_i / f_local
                t_off_unit = (1 + self.cfg.task.result_ratio) * D_i / R_i + C_i / f_alloc
                alpha_kkt = t_local_full / (t_local_full + t_off_unit)
                alpha_kkt = np.clip(alpha_kkt, 0.01, 0.99)
                alphas[i] = np.clip(alpha_base * alpha_kkt, 0.0, 1.0)

        return alphas

    def _select_users_by_cluster_weights(self, agent_actions, assignment, uav_positions):
        """DRL-controlled user selection using cluster priority weights."""
        N = self.cfg.num_users
        K = self.cfg.num_uavs
        nc = self.cfg.num_clusters
        max_served = self.cfg.n_subcarriers * self.cfg.users_per_sub
        served = np.zeros(N, dtype=bool)

        for k in range(K):
            act = agent_actions[k]
            # Parse cluster weights: first nc-1 elements + reference=0
            raw = np.zeros(nc)
            for c in range(min(nc - 1, len(act) - 1)):
                raw[c] = float(act[c])
            # Last cluster = 0 (reference)
            exp_r = np.exp(raw - raw.max())
            cw = exp_r / exp_r.sum()

            candidates = [i for i in range(N)
                          if assignment[i] == k and self.task_queue_active[i]]
            if not candidates:
                continue

            uav_pos = uav_positions[k]
            scores = {}
            for i in candidates:
                cid = self.user_cluster_ids[i]
                h_i = self.channel.path_loss(
                    np.linalg.norm(np.array(uav_pos[:2]) - self.user_positions[i]))
                scores[i] = cw[min(cid, nc - 1)] * h_i

            sorted_users = sorted(candidates, key=lambda i: scores[i], reverse=True)
            for i in sorted_users[:max_served]:
                served[i] = True

        return served

    def _update_task_outcome(self, i, user_delay, user_qualities):
        """Update task queue and quality metric for user i."""
        _x = 5.0 * (user_delay - self.cfg.task.t_max)
        sig = 1.0 / (1.0 + np.exp(-np.clip(_x, -20, 20)))
        user_qualities.append(1.0 - sig)
        if user_delay <= self.cfg.task.t_max:
            self.task_queue_active[i] = False
            self.task_queue_age[i] = 0
        else:
            self.task_queue_age[i] += 1
            if self.task_queue_age[i] >= self.cfg.max_queue_age:
                self.task_queue_active[i] = False
                self.task_queue_age[i] = 0

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        N = self.cfg.num_users
        K = self.cfg.num_uavs

        # Parse per-agent actions
        agent_actions = []
        for k in range(K):
            a = action[k * self.agent_action_dim:(k + 1) * self.agent_action_dim]
            agent_actions.append(a)

        # Hovering energy
        e_hover = self.cfg.uav.c1 * self.cfg.uav.dt
        for k in range(K):
            self.batteries[k] -= e_hover

        # User mobility
        if self.cfg.user_mobility:
            dt = self.cfg.uav.dt
            for i in range(N):
                angle = self.rng.uniform(0, 2 * math.pi)
                step_d = self.cfg.user_speed * dt
                self.user_positions[i, 0] += step_d * math.cos(angle)
                self.user_positions[i, 1] += step_d * math.sin(angle)
            self.user_positions = np.clip(self.user_positions, 0, self.cfg.area_size)

        # Assignment
        assignment = self._assign_users()

        # Grouped NOMA pipeline
        powers_w = self._compute_powers(assignment)
        uav_positions = [uav.position for uav in self.uavs]
        served = self._select_users_by_cluster_weights(
            agent_actions, assignment, uav_positions)
        rates, _ = self.channel.compute_grouped_noma_rates(
            uav_positions, self.user_positions, assignment,
            self.task_queue_active & served,
            n_sub=self.cfg.n_subcarriers,
            users_per_sub=self.cfg.users_per_sub,
            sic_eps=0.02)
        alphas = self._compute_alphas(agent_actions, assignment, rates, served)

        # Delay computation
        total_delay, violations, tasks_served = 0.0, 0, 0
        user_qualities = []

        for k in range(K):
            served_k = [i for i in range(N)
                        if assignment[i] == k and self.task_queue_active[i] and served[i]]
            unserved_k = [i for i in range(N)
                          if assignment[i] == k and self.task_queue_active[i] and not served[i]]

            n_offloading = sum(1 for i in served_k if alphas[i] > 0.01)
            f_alloc = self.cfg.f_per_uav / max(n_offloading, 1)
            f_alloc = max(f_alloc, 1e3)

            for i in served_k:
                tasks_served += 1
                alpha = alphas[i]
                C_local = (1 - alpha) * self.task_queue_comp[i]
                t_local = local_delay(C_local, self.cfg.task.f_local)
                D_off = alpha * self.task_queue_data[i]
                C_off = alpha * self.task_queue_comp[i]
                t_off = offload_delay(D_off, C_off, rates[i], f_alloc)
                user_delay = max(t_local, t_off)
                total_delay += user_delay
                self._update_task_outcome(i, user_delay, user_qualities)
                if user_delay > self.cfg.task.t_max:
                    violations += 1

            for i in unserved_k:
                tasks_served += 1
                user_delay = local_delay(self.task_queue_comp[i], self.cfg.task.f_local)
                total_delay += user_delay
                self._update_task_outcome(i, user_delay, user_qualities)
                if user_delay > self.cfg.task.t_max:
                    violations += 1

        # Reward
        safe_tasks = max(tasks_served, 1)
        met = tasks_served - violations
        reward = met / safe_tasks

        # Jain fairness
        if len(user_qualities) >= 2:
            q = np.array(user_qualities)
            jain = (q.sum() ** 2) / (len(q) * (q ** 2).sum() + 1e-10)
        else:
            jain = 1.0

        self.ep_tasks += tasks_served
        self.ep_violations += violations
        self.ep_delay += total_delay

        # Advance + new tasks
        self.slot += 1
        terminated = self.slot >= self.cfg.num_slots
        nc = self.cfg.num_clusters

        if not terminated:
            new_tasks = self.task_gen.generate(N)
            if self.cfg.random_burst:
                if self.slot >= self.next_burst_switch:
                    self.burst_cluster = self.rng.integers(0, nc)
                    self.next_burst_switch = self.slot + self.rng.integers(
                        self.cfg.burst_period_min, self.cfg.burst_period_max + 1)
                for i in range(N):
                    cid = self.user_cluster_ids[i]
                    prob = self.cfg.burst_rate if cid == self.burst_cluster else self.cfg.cool_rate
                    new_tasks["has_task"][i] = self.rng.random() < prob
                    if not new_tasks["has_task"][i]:
                        new_tasks["data_bits"][i] = 0
                        new_tasks["comp_cycles"][i] = 0
            elif self.cfg.dynamic_burst:
                burst_idx = (self.slot // self.cfg.burst_cycle) % nc
                for i in range(N):
                    cid = self.user_cluster_ids[i]
                    prob = self.cfg.burst_rate if cid == burst_idx else self.cfg.cool_rate
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

        # Mean UAV separation
        if K >= 2:
            seps = []
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    seps.append(np.linalg.norm(
                        np.array([self.uavs[k1].x - self.uavs[k2].x,
                                  self.uavs[k1].y - self.uavs[k2].y])))
            uav_sep = np.mean(seps)
        else:
            uav_sep = 0.0

        info = {
            "episode_tasks_served": self.ep_tasks,
            "episode_deadline_violations": self.ep_violations,
            "slot_tasks": tasks_served,
            "slot_violations": violations,
            "jain_index": jain,
            "slot_delay": total_delay,
            "uav_dist": uav_sep,
            "uav_separation": uav_sep,
            "deadline_rate": met / max(safe_tasks, 1),
        }

        return self._get_obs(), float(reward), terminated, False, info

    def _get_obs(self):
        """Per-agent observations concatenated. K-generic."""
        K = self.cfg.num_uavs
        nc = self.cfg.num_clusters
        L = self.cfg.area_size

        # Cluster loads
        cluster_loads = np.zeros(nc)
        for i in range(self.cfg.num_users):
            if self.task_queue_active[i]:
                cid = self.user_cluster_ids[i]
                cluster_loads[min(cid, nc - 1)] += 1
        max_load = max(cluster_loads.max(), 1)
        cluster_loads /= max_load

        obs_parts = []
        for k in range(K):
            own_xy = np.array([self.uavs[k].x / L, self.uavs[k].y / L])

            # Mean position of all OTHER UAVs (K-generic)
            if K > 1:
                other_xs = [self.uavs[j].x / L for j in range(K) if j != k]
                other_ys = [self.uavs[j].y / L for j in range(K) if j != k]
                other_xy = np.array([np.mean(other_xs), np.mean(other_ys)])
            else:
                other_xy = own_xy.copy()

            batt = np.array([max(self.batteries[k] / self.cfg.battery_budget, 0)])
            time_frac = np.array([1.0 - self.slot / self.cfg.num_slots])
            obs_parts.append(np.concatenate([
                own_xy, other_xy, cluster_loads[:nc], batt, time_frac
            ]))

        return np.concatenate(obs_parts).astype(np.float32)
