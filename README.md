# V2X MEC Offloading Thesis

Event-Triggered Periodic Task Offloading with Phase-Aligned Scheduling for V2X Cooperative Perception

面向V2X协作感知的事件触发型周期性任务卸载与相位对齐调度机制

## Overview

This project implements simulation framework for research on computation offloading in V2X (Vehicle-to-Everything) scenarios with Mobile Edge Computing (MEC).

## Project Structure

```
V2X_MEC_Offloading_Thesis/
├── docs/                    # Technical design documents
│   ├── 00_研究概述与目录.md
│   ├── 01_系统模型.md
│   ├── 02_问题形式化与理论分析.md
│   ├── 03_算法设计_相位对齐调度.md
│   ├── 04_算法设计_任务负载预测.md
│   ├── 05_算法设计_Lyapunov优化.md
│   ├── 06_算法设计_资源分配与服务迁移.md
│   ├── 07_仿真实验设计.md
│   ├── 08_基线算法实现.md
│   └── 09_论文写作指南.md
├── simulation/              # Simulation framework
│   ├── core/               # Core simulation components
│   ├── algorithms/         # Proposed algorithms
│   ├── baselines/          # Baseline algorithms
│   ├── generators/         # Data generators
│   ├── metrics/            # Evaluation metrics
│   ├── experiments/        # Experiment runners
│   └── utils/              # Utility functions
└── README.md
```

## Key Features

- Phase-aligned scheduling for periodic tasks
- Lyapunov-based online optimization
- Task load prediction (kinematic + learning-based)
- Proactive service migration
- Comprehensive baseline implementations

## Requirements

- Python >= 3.8
- NumPy
- SciPy
- Matplotlib
- (Additional dependencies to be added)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd V2X_MEC_Offloading_Thesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run simulation (example)
python -m simulation.experiments.main
```

## Documentation

Detailed technical documentation is available in the `docs/` directory (in Chinese).

## License

This project is for academic research purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{author2024v2x,
  title={Event-Triggered Periodic Task Offloading with Phase-Aligned Scheduling for V2X Cooperative Perception},
  author={Author},
  journal={TBD},
  year={2024}
}
```

# 进入仿真目录
cd /home/ggmx/my_workspace/my_project/V2X_MEC_Offloading_Thesis

# 测试 task.py (已完成)
python3 -m simulation.core.task

# 测试 vehicle.py
python3 -m simulation.core.vehicle

# 测试 rsu.py
python3 -m simulation.core.rsu

# 测试 channel.py
python3 -m simulation.core.channel