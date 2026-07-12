import json
from pathlib import Path
import re

import pytest

from uav_mec.generate_figures_v2 import generate_assets, method_mean_ci


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_canonical(root: Path, gates_pass: bool = True) -> dict:
    index = {
        "runs": {
            name: f"runs/{name}"
            for name in ("main", "pairing", "resource", "exact")
        },
        "analysis": "analysis",
        "suites": {},
    }
    write_json(root / "index.json", index)
    summaries = {
        "main": {
            "seeds": [1, 2, 3, 4],
            "methods": {
                "full": {"on_time_rate_by_seed": [0.80, 0.82, 0.84, 0.86]},
                "exchange_stable": {
                    "on_time_rate_by_seed": [0.70, 0.72, 0.74, 0.76]
                },
            },
        },
        "pairing": {
            "seeds": [1, 2, 3, 4],
            "methods": {
                "intra_bilateral": {"on_time_rate_by_seed": [0.60] * 4},
                "cross_bilateral": {"on_time_rate_by_seed": [0.65] * 4},
                "intra_global": {"on_time_rate_by_seed": [0.70] * 4},
                "full": {"on_time_rate_by_seed": [0.80] * 4},
            },
        },
        "resource": {
            "seeds": [1, 2, 3, 4],
            "methods": {
                "full_fixed_half": {"on_time_rate_by_seed": [0.55] * 4},
                "full_crossing": {"on_time_rate_by_seed": [0.70] * 4},
                "full": {"on_time_rate_by_seed": [0.80] * 4},
            },
        },
        "exact": {"optimal_slot_fraction": 0.92, "mean_completion_gap": 0.08},
    }
    for name, summary in summaries.items():
        write_json(root / f"runs/{name}/summary.json", summary)
    statistics = {
        "innovation1_cross": {
            "mean_difference_pp": 10.0,
            "ci_low_pp": 8.0,
            "ci_high_pp": 12.0,
            "p_holm": 0.01,
            "rank_biserial": 1.0,
        },
        "innovation1_acceptance": {
            "mean_difference_pp": 15.0,
            "ci_low_pp": 13.0,
            "ci_high_pp": 17.0,
            "p_holm": 0.01,
            "rank_biserial": 1.0,
        },
        "innovation2_deadline": {
            "mean_difference_pp": 10.0,
            "ci_low_pp": 9.0,
            "ci_high_pp": 11.0,
            "p_holm": 0.01,
            "rank_biserial": 1.0,
        },
        "baseline_exchange": {
            "mean_difference_pp": 10.0,
            "ci_low_pp": 8.0,
            "ci_high_pp": 12.0,
            "p_holm": 0.01,
            "rank_biserial": 1.0,
        },
    }
    write_json(root / "analysis/statistics.json", statistics)
    write_json(
        root / "analysis/metrics.json",
        {
            "main": {
                "full": {
                    "on_time_rate": 0.83,
                    "drop_rate": 0.17,
                    "mean_completed_delay_s": 0.71,
                    "jain_user_on_time": 0.96,
                    "cpu_utilization": 0.72,
                    "runtime_p50_s": 0.08,
                    "runtime_p95_s": 0.12,
                }
            }
        },
    )
    write_json(
        root / "analysis/acceptance.json",
        {
            "all_gates_pass": gates_pass,
            "validation": {"main": {"sic_violations": 0}},
        },
    )
    return summaries


def test_error_bar_is_bootstrap_ci_and_macros_match_statistics(tmp_path: Path):
    root = tmp_path / "canonical"
    summaries = make_canonical(root)
    figures = tmp_path / "figures"
    macros = tmp_path / "generated_results.tex"
    statistics_table = tmp_path / "generated_statistics.tex"
    robustness = tmp_path / "generated_robustness.tex"

    generate_assets(root, figures, macros, statistics_table, robustness)

    mean, low, high = method_mean_ci(summaries["main"], "full")
    assert mean == pytest.approx(83.0)
    assert low < mean < high
    text = macros.read_text(encoding="utf-8")
    assert "\\newcommand{\\InnovationOneCrossGainPP}{10.00}" in text
    assert "\\newcommand{\\ExactOptimalPercent}{92.00\\%}" in text
    assert "\\newcommand{\\FullRuntimePNinetyFiveMs}{120.00}" in text
    assert not re.search(r"\\newcommand\{\\[A-Za-z]*\d", text)
    assert "innovation1\\_cross" in statistics_table.read_text(encoding="utf-8")
    assert robustness.exists()
    assert (figures / "main_on_time.pdf").exists()
    assert (figures / "pairing_ablation.pdf").exists()
    assert (figures / "resource_ablation.pdf").exists()


def test_generator_refuses_failed_evidence_gates(tmp_path: Path):
    root = tmp_path / "canonical"
    make_canonical(root, gates_pass=False)

    with pytest.raises(RuntimeError, match="failed evidence gates"):
        generate_assets(
            root,
            tmp_path / "figures",
            tmp_path / "results.tex",
            tmp_path / "statistics.tex",
            tmp_path / "robustness.tex",
        )
