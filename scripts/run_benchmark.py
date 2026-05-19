import hydra
import json
import optuna
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from datetime import datetime, timezone
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from time import perf_counter

# Force CPU to avoid driver warnings
torch.set_default_device("cpu")


def seed_everything(seed: int):
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    optuna.samplers.RandomSampler(seed=seed)  # warms up optuna's internal state


def make_objective(test_func):
    """
    Dynamically builds the objective using the test function's specific bounds.
    test_func.bounds is a PyTorch tensor of shape (2, dim).
    Row 0 is lower bounds, Row 1 is upper bounds.
    """
    bounds = test_func.bounds
    dim = bounds.shape[1]

    def objective(trial):
        x = torch.tensor(
            [
                trial.suggest_float(
                    f"x{i}", 
                    bounds[0, i].item(), 
                    bounds[1, i].item()
                ) for i in range(dim)
            ]
        )
        x = x.clamp(bounds[0], bounds[1])
        return test_func(x.unsqueeze(0)).item()

    return objective


def compute_best_so_far(study, global_min):
    best_so_far = []
    current_best = float("inf")

    for trial in study.trials:
        current_best = min(current_best, trial.value)
        best_so_far.append(current_best - global_min)

    return best_so_far


def save_results(cfg: DictConfig, regrets: list[float], trial_times: list[float]):
    """Save regrets, per-trial times, and full config to a JSON file in the
    Hydra output directory so aggregate_and_plot.py can find them later."""
    sampler_name = cfg.sampler.sampler_name
    seed = cfg.get("seed", 0)
 
    results = {
        "sampler_name": sampler_name,
        "benchmark_name": cfg.benchmark.name,
        "seed": seed,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "n_trials": cfg.experiment.n_trials,
        "n_startup_trials": cfg.experiment.n_startup_trials,
        "mc_budget": cfg.experiment.mc_budget,
        "regrets": regrets,
        "trial_times_seconds": trial_times,
        "total_time_seconds": sum(trial_times),
    }
 
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    output_path = run_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {output_path.resolve()}")


def plot_regret(cfg: DictConfig, regrets: list[float]):
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        sampler_name = cfg.sampler.sampler_name
        filename = f"{cfg.benchmark.name}_{sampler_name}.png"
        output_path = figures_dir / filename

        plt.figure(figsize=(8, 5))

        plt.plot(
            regrets,
            label=cfg.sampler._target_.split(".")[-1],
            linewidth=2,
        )

        plt.axvline(
            x=cfg.experiment.n_startup_trials,
            color="gray",
            linestyle="--",
            label="End of Random Startup",
        )

        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Best-so-far Regret")
        plt.title(f"{cfg.benchmark.name} Benchmark - {sampler_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Plot saved to: {output_path.resolve()}")
        
    except Exception as e:
        print(f"[WARNING] Could not save per-run plot: {e}")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):

    if cfg.debug:
        print(OmegaConf.to_yaml(cfg))
        return

    seed = cfg.get("seed", 0)
    seed_everything(seed)

    sampler = hydra.utils.instantiate(cfg.sampler)
    test_func = hydra.utils.instantiate(cfg.benchmark.function)

    sampler_name = cfg.sampler.sampler_name

    print(f"[seed={seed}] Running {cfg.benchmark.name} (dim={test_func.bounds.shape[1]}) with {sampler_name}")

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    objective = make_objective(test_func)
    trial_times = []

    def timed_objective(trial):
        t0 = perf_counter()
        value = objective(trial)
        trial_times.append(perf_counter() - t0)
        return value
 
    study.optimize(timed_objective, n_trials=cfg.experiment.n_trials)
 
    print(f"Study complete. Best value: {study.best_value:.4f}")
 
    regrets = compute_best_so_far(study, cfg.benchmark.global_min)
 
    save_results(cfg, regrets, trial_times)
    plot_regret(cfg, regrets)

if __name__ == "__main__":
    main()