"""
aggregate_and_plot.py

Scans all outputs/ subdirectories for results.json files, groups them by
benchmark and sampler, then produces one publication-quality plot per
benchmark with mean regret curves and shaded ±1 std confidence bands.

Usage:
    uv run scripts/aggregate_and_plot.py
    uv run scripts/aggregate_and_plot.py outputs_dir=multirun/2026-05-12/10-00-00
"""

import json
import hydra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv()

# Consistent colors and display order across all plots
SAMPLER_STYLE = {
    "VanillaOptunaBO":      {"color": "#4878CF", "label": "Vanilla BO"},
    "NaiveMarginalOptunaBO":{"color": "#e8a838", "label": "Naive Marginal BO"},
    "OrthoBO":              {"color": "#2ca02c", "label": "OrthoBO (ours)"},
}
SAMPLER_ORDER = ["VanillaOptunaBO", "NaiveMarginalOptunaBO", "OrthoBO"]


def load_all_results(outputs_dir: Path) -> list[dict]:
    """Recursively find and load every results.json under outputs_dir."""
    result_files = list(outputs_dir.rglob("results.json"))
    if not result_files:
        raise FileNotFoundError(f"No results.json files found under {outputs_dir}")

    print(f"Found {len(result_files)} result file(s) under {outputs_dir}")
    records = []
    for path in result_files:
        try:
            records.append(json.loads(path.read_text()))
        except Exception as e:
            print(f"  [WARN] Could not load {path}: {e}")
    return records


def group_results(records: list[dict]) -> dict:
    """
    Returns a nested dict:
        grouped[benchmark_name][sampler_name] = list of regret arrays
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r["benchmark_name"]][r["sampler_name"]].append(r["regrets"])
    return grouped


def pad_and_stack(regret_lists: list[list[float]]) -> np.ndarray:
    """
    Stack regret curves of potentially different lengths into a 2D array.
    Shorter curves are forward-filled with their last value so all rows
    have the same length. Shape: (n_seeds, max_trials).
    """
    max_len = max(len(r) for r in regret_lists)
    padded = []
    for r in regret_lists:
        arr = np.array(r, dtype=float)
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len - len(arr), arr[-1])])
        padded.append(arr)
    return np.stack(padded, axis=0)


def plot_benchmark(benchmark_name: str, sampler_data: dict, figures_dir: Path):
    """
    sampler_data: {sampler_name: list_of_regret_curves}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for sampler_name in SAMPLER_ORDER:
        if sampler_name not in sampler_data:
            continue

        style = SAMPLER_STYLE.get(
            sampler_name,
            {"color": "gray", "label": sampler_name},
        )

        matrix = pad_and_stack(sampler_data[sampler_name])  # (seeds, trials)
        n_seeds = matrix.shape[0]
        mean = matrix.mean(axis=0)
        std  = matrix.std(axis=0)
        xs   = np.arange(1, len(mean) + 1)

        ax.plot(xs, mean, label=f"{style['label']} (n={n_seeds})",
                color=style["color"], linewidth=2)
        ax.fill_between(xs, mean - std, mean + std,
                        color=style["color"], alpha=0.15)

    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best-so-far Regret", fontsize=12)
    ax.set_title(f"{benchmark_name} — mean ± 1 std across seeds", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"{benchmark_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary(grouped: dict):
    """Print a quick table showing how many seeds were collected per cell."""
    print("\n── Seed count summary ──────────────────────────────────")
    for benchmark, samplers in sorted(grouped.items()):
        print(f"  {benchmark}")
        for sampler, curves in sorted(samplers.items()):
            label = SAMPLER_STYLE.get(sampler, {}).get("label", sampler)
            lengths = [len(c) for c in curves]
            print(f"    {label:30s}  seeds={len(curves)}  trials={lengths}")
    print("────────────────────────────────────────────────────────\n")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):
    outputs_dir = Path(cfg.paths.root)
    figures_dir = Path(cfg.paths.figures)

    print(f"Scanning: {outputs_dir}")

    records = load_all_results(outputs_dir)
    grouped = group_results(records)

    print_summary(grouped)

    print("Generating plots...")
    for benchmark_name, sampler_data in sorted(grouped.items()):
        plot_benchmark(benchmark_name, sampler_data, figures_dir)

    print("Done.")


if __name__ == "__main__":
    main()