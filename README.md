# V2X MEC Offloading Thesis

Event-Triggered Periodic Task Offloading with Phase-Aligned Scheduling for V2X Cooperative Perception

面向V2X协作感知的事件触发型周期性任务卸载与相位对齐调度机制 (PAPO)

## Overview

This project implements a simulation framework for research on computation offloading in V2X (Vehicle-to-Everything) scenarios with Mobile Edge Computing (MEC). The framework focuses on Collective Perception Message (CPM) processing based on ETSI TS 103 324 standard.

### Key Contributions

1. **Phase-Aligned Scheduling**: CPA (Centralized) and DPA (Distributed, 2-competitive) algorithms for reducing task collision
2. **Lyapunov-based Online Optimization**: Drift-Plus-Penalty framework balancing AoI, energy, and deadline
3. **Task Load Prediction**: Kinematic and historical prediction for proactive resource allocation
4. **KKT-based Resource Allocation**: Optimal CPU frequency allocation with closed-form solution

## Project Structure

```
V2X_MEC_Offloading_Thesis/
├── config/
│   └── parameters.yaml          # Simulation parameters
├── simulation/
│   ├── core/                    # Core simulation components
│   │   ├── vehicle.py           # Vehicle class
│   │   ├── rsu.py               # RSU class
│   │   ├── task.py              # CPM task class
│   │   └── channel.py           # Channel model
│   ├── algorithms/              # Proposed algorithms
│   │   ├── cpa.py               # Centralized Phase Allocation
│   │   ├── dpa.py               # Distributed Phase Adjustment
│   │   ├── lyapunov.py          # Lyapunov optimization
│   │   ├── resource_alloc.py    # KKT resource allocation
│   │   └── predictor.py         # Task prediction
│   ├── baselines/               # Baseline algorithms
│   │   ├── local_only.py        # Local computing only
│   │   ├── full_offload.py      # Full offloading
│   │   ├── greedy.py            # Greedy offloading
│   │   ├── random_offload.py    # Random offloading
│   │   ├── threshold.py         # Threshold-based offloading
│   │   └── no_phase_align.py    # Without phase alignment
│   ├── generators/              # Data generators
│   │   ├── trajectory.py        # Vehicle trajectory (3GPP TR 37.885)
│   │   ├── task_generator.py    # CPM task generation (ETSI TS 103 324)
│   │   └── channel_generator.py # Channel coefficient generation
│   ├── metrics/                 # Evaluation metrics
│   │   ├── aoi.py               # Age of Information
│   │   ├── energy.py            # Energy consumption
│   │   ├── deadline.py          # Deadline satisfaction
│   │   └── fairness.py          # Jain Fairness Index
│   ├── experiments/             # Experiment runners
│   │   ├── simulator.py         # Main simulation engine
│   │   └── runner.py            # Batch experiment runner
│   └── utils/                   # Utility functions
│       ├── logger.py            # Logging system
│       └── visualization.py     # Visualization tools
├── docs/                        # Technical design documents (Chinese)
├── main.py                      # Main entry point
└── README.md
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.5
- PyYAML >= 6.0

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd V2X_MEC_Offloading_Thesis

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install numpy scipy matplotlib pyyaml
```

## Usage

### Quick Test
```bash
# Run a quick simulation (10 vehicles, 10 seconds)
python3 main.py --mode quick
```

### Standard Simulation
```bash
# Run standard simulation (50 vehicles, 100 seconds)
python3 main.py --mode standard
```

### View Help
```bash
python3 main.py --help
```

### Module Testing
```bash
# Test core modules
python3 -m simulation.core.task
python3 -m simulation.core.vehicle
python3 -m simulation.core.rsu
python3 -m simulation.core.channel

# Test generators
python3 -m simulation.generators.trajectory
python3 -m simulation.generators.task_generator

# Test algorithms
python3 -m simulation.algorithms.cpa
python3 -m simulation.algorithms.dpa
python3 -m simulation.algorithms.lyapunov
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| T_check | 0.1 s | CPM check period |
| T_max | 1.0 s | Maximum CPM interval |
| Deadline | 80 ms | Task deadline |
| RSU CPU | 20 GHz | RSU computing capacity |
| Bandwidth | 20 MHz | Wireless bandwidth |
| Coverage | 400 m | RSU coverage radius |

## Algorithm Overview

### PAPO (Phase-Aligned Periodic Offloading)
- **CPA**: φ_i = (i-1) × T_check / N (optimal for known N)
- **DPA**: Insert at maximum gap midpoint (2-competitive guarantee)
- **Lyapunov**: min{V×cost + Q×drift} with virtual queues
- **KKT**: f_j* = √(w_j×C_j) / Σ√(w_i×C_i) × F_max

## Documentation

Detailed technical documentation is available in the `docs/` directory (in Chinese):
- [00_研究路线图.md](docs/00_研究路线图.md) - Research roadmap
- [01_系统模型.md](docs/01_系统模型.md) - System model
- [02_问题形式化与理论分析.md](docs/02_问题形式化与理论分析.md) - Problem formulation
- [03_算法设计_相位对齐调度.md](docs/03_算法设计_相位对齐调度.md) - Phase alignment
- [05_算法设计_Lyapunov优化.md](docs/05_算法设计_Lyapunov优化.md) - Lyapunov optimization

## Current Status

**Progress: 70%**

- [x] Theoretical analysis and proofs
- [x] Core simulation framework
- [x] Algorithm implementation
- [x] Baseline implementations
- [ ] Parameter tuning and validation
- [ ] Experiment execution
- [ ] Paper writing

## License

This project is for academic research purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{author2024v2x,
  title={Event-Triggered Periodic Task Offloading with Phase-Aligned Scheduling for V2X Cooperative Perception},
  author={Author},
  journal={IEEE Transactions on Vehicular Technology},
  year={2024}
}
```