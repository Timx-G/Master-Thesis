"""
Train MAPPO for 2000 episodes and save episode-by-episode reward history.
Run from project root: python run_mappo_convergence.py
"""
import json, time
import numpy as np
from pathlib import Path

from uav_mec.model import ScenarioConfig
from uav_mec.run_drl_convergence_and_mappo import train_mappo, evaluate_mappo

cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True, user_mobility=True)
train_seeds = list(range(100, 322))
test_seeds  = list(range(42, 62))

print("Training MAPPO 2000 episodes ...")
t0 = time.time()
agent, history = train_mappo(cfg, train_seeds, n_episodes=2000, verbose=True)
elapsed = time.time() - t0
print(f"Training done in {elapsed/60:.1f} min")

print("Zero-shot evaluation on test seeds ...")
test_dl = evaluate_mappo(agent, cfg, test_seeds)
test_mean = float(np.mean(test_dl))
test_std  = float(np.std(test_dl, ddof=1))
print(f"Zero-shot: {test_mean:.2f} ± {test_std:.2f}%")

# Save
out = {
    "history": [round(x, 3) for x in history],          # raw per-episode
    "rolling50": [round(x, 3) for x in
                  np.convolve(history, np.ones(50)/50, mode='valid').tolist()],
    "test_mean": round(test_mean, 3),
    "test_std":  round(test_std, 3),
    "test_seeds": test_dl,
    "n_episodes": len(history),
    "elapsed_s": round(elapsed, 1),
}
Path("results/mappo_convergence_history.json").write_text(json.dumps(out, indent=2))
print("Saved: results/mappo_convergence_history.json")
print(f"history len={len(history)}, rolling50 len={len(out['rolling50'])}")
