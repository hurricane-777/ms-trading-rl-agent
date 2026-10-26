"""Read Draft 2 final_results.json. Every displayed number comes from that file."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_TOP = (
    "config",
    "environment",
    "data_source",
    "variants",
    "seeds",
    "test_used",
    "runs",
    "baselines",
)
METRIC_KEYS = (
    "total_return",
    "sharpe",
    "max_drawdown",
    "n_trades",
    "transaction_costs",
    "final_value",
    "avg_exposure",
)
# (split label, weights label) -> key under run["metrics"]
METRIC_BLOCK = {
    ("Train", "selected"): "train",
    ("Validation", "selected"): "val",
    ("Validation", "final"): "val_final",
    ("Test", "selected"): "test",
    ("Test", "final"): "test_final",
}
BASELINE_SPLIT = {
    "Train": "train",
    "Validation": "validation",
    "Test": "test",
}
RUN_FIELDS = ("variant", "seed", "episodes", "selected_episode", "selection_score", "minutes", "metrics")


def artifact_root() -> Path:
    override = os.environ.get("TRADING_AGENT_ROOT")
    if override:
        return Path(override)
    return ROOT


def results_path(root: Path | None = None) -> Path:
    return (root or artifact_root()) / "final_results.json"


def run_name(variant: str, seed: int) -> str:
    return f"{variant.lower().replace(' + ', '_').replace(' ', '_')}_seed{int(seed)}"


def checkpoint_path(variant: str, seed: int, weights: str = "selected", root: Path | None = None) -> Path:
    if weights not in ("selected", "final"):
        raise ValueError(f"unknown weights {weights!r}")
    return (root or artifact_root()) / "runs" / f"{run_name(variant, seed)}_{weights}.pt"


def selected_checkpoint(variant: str, seed: int, root: Path | None = None) -> Path:
    return checkpoint_path(variant, seed, "selected", root)


def load_results(path: Path | None = None) -> dict:
    path = path or results_path()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_TOP if key not in data]
    if missing:
        raise ValueError(f"{path} is missing keys: {', '.join(missing)}")
    if len(data["runs"]) != len(data["variants"]) * len(data["seeds"]):
        raise ValueError(
            f"{path} has {len(data['runs'])} runs for "
            f"{len(data['variants'])} variants and {len(data['seeds'])} seeds"
        )
    for run in data["runs"]:
        absent = [key for key in RUN_FIELDS if key not in run]
        if absent:
            raise ValueError(f"run {run.get('variant')} seed {run.get('seed')} missing {absent}")
        for block in ("train", "val", "val_final"):
            _require_metrics(run, block)
        if data["test_used"]:
            for block in ("test", "test_final"):
                _require_metrics(run, block)
    return data


def _require_metrics(run: dict, block: str) -> None:
    metrics = run["metrics"].get(block)
    if not isinstance(metrics, dict):
        raise ValueError(f"{run['variant']} seed {run['seed']} has no metrics.{block}")
    absent = [key for key in METRIC_KEYS if key not in metrics]
    if absent:
        raise ValueError(f"{run['variant']} seed {run['seed']} metrics.{block} missing {absent}")


def find_run(data: dict, variant: str, seed: int) -> dict:
    for run in data["runs"]:
        if run["variant"] == variant and int(run["seed"]) == int(seed):
            return run
    raise KeyError(f"no saved run for {variant} seed {seed}")


def metric_block(run: dict, split: str, weights: str) -> dict:
    key = METRIC_BLOCK[(split, weights)]
    block = run["metrics"].get(key)
    if block is None:
        raise KeyError(f"{run['variant']} seed {run['seed']} has no {key}")
    return block


def ordered_runs(data: dict) -> list[dict]:
    order = {name: i for i, name in enumerate(data["variants"])}
    return sorted(data["runs"], key=lambda run: (order.get(run["variant"], 99), int(run["seed"])))


def format_metric(key: str, value: float) -> str:
    return _fmt(key, float(value))


def _fmt(key: str, value: float) -> str:
    if key in ("total_return", "max_drawdown"):
        return f"{100.0 * value:.1f}%"
    if key == "avg_exposure":
        return f"{100.0 * value:.0f}%"
    if key == "sharpe":
        return f"{value:.2f}"
    if key == "n_trades":
        return f"{value:.0f}"
    if key in ("transaction_costs", "final_value"):
        return f"${value:,.0f}" if key == "transaction_costs" else f"${value:,.2f}"
    return str(value)


def mean_std_text(values: list[float], key: str) -> str:
    """Format the mean of stored numbers. Sample std when there is more than one."""
    vals = [float(v) for v in values]
    if len(vals) == 1:
        return _fmt(key, vals[0])
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return f"{_fmt(key, mean)} ± {_fmt(key, math.sqrt(var)).lstrip('$')}"


def per_run_rows(data: dict, split: str, weights: str) -> list[dict]:
    rows = []
    for run in ordered_runs(data):
        metrics = metric_block(run, split, weights)
        row = {
            "Variant": run["variant"],
            "Seed": int(run["seed"]),
            "Episodes": run["episodes"],
            "Selected episode": run["selected_episode"],
            "Selection score": run["selection_score"],
            "Minutes": run["minutes"],
        }
        for key in METRIC_KEYS:
            row[key] = metrics[key]
        rows.append(row)
    return rows


def summary_rows(data: dict, split: str, weights: str) -> list[dict]:
    rows = []
    for variant in data["variants"]:
        group = [metric_block(run, split, weights) for run in data["runs"] if run["variant"] == variant]
        row = {"Variant": variant, "Seeds": len(group)}
        for key in METRIC_KEYS:
            row[key] = mean_std_text([item[key] for item in group], key)
        rows.append(row)
    return rows


def _baseline_entries(value):
    if isinstance(value, dict) and "total_return" in value:
        return [value]
    if isinstance(value, list):
        return value
    return []


def baseline_rows(data: dict, split: str) -> list[dict]:
    block = data["baselines"].get(BASELINE_SPLIT[split])
    if not block:
        return []
    rows = []
    for name, value in block.items():
        entries = _baseline_entries(value)
        if not entries:
            continue
        row = {"Baseline": name, "Rows in file": len(entries)}
        for key in METRIC_KEYS:
            if key not in entries[0]:
                continue
            row[key] = mean_std_text([item[key] for item in entries], key)
        rows.append(row)
    return rows


def figures(root: Path | None = None) -> list[Path]:
    folder = (root or artifact_root()) / "figures"
    names = [
        "fig1_data_splits.png",
        "fig2_training_curves.png",
        "fig3_validation_sharpe.png",
        "fig4_test_equity.png",
        "fig5_test_trades.png",
        "fig6_ablation_test.png",
    ]
    return [folder / name for name in names if (folder / name).is_file()]
