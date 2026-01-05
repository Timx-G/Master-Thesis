"""
实验模块
文件: simulation/experiments/__init__.py
"""

from simulation.experiments.runner import ExperimentConfig, ExperimentRunner, print_comparison_table
from simulation.experiments.simulator import (
    PhaseAllocMethod,
    Simulator,
    SimulatorConfig,
    SlotResult,
)

__all__ = [
    # Simulator
    "PhaseAllocMethod",
    "SimulatorConfig",
    "SlotResult",
    "Simulator",
    # Runner
    "ExperimentConfig",
    "ExperimentRunner",
    "print_comparison_table",
]
