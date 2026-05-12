import hydra
import optuna
import torch
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from botorch.test_functions import Hartmann

load_dotenv()

# Force CPU to avoid driver warnings
torch.set_default_device("cpu")


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
        return test_func(x.unsqueeze(0)).item()

    return objective


def compute_best_so_far(study, global_min):
    best_so_far = []
    current_best = float("inf")

    for trial in study.trials:
        current_best = min(current_best, trial.value)
        best_so_far.append(current_best - global_min)

    return best_so_far


def plot_regret(cfg: DictConfig, regrets: list[float]):
    figures_dir = Path(cfg.paths.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sampler_name = cfg.sampler._target_.split(".")[-1]
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


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):

    if cfg.debug:
        print(OmegaConf.to_yaml(cfg))
        return

    sampler = hydra.utils.instantiate(cfg.sampler)

    test_func = hydra.utils.instantiate(cfg.benchmark.function)

    print(f"Running {cfg.benchmark.name} (dim={test_func.bounds.shape[1]})")

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        make_objective(test_func),
        n_trials=cfg.experiment.n_trials,
    )

    print(f"Study complete. Best value: {study.best_value:.4f}")

    regrets = compute_best_so_far(
        study,
        cfg.benchmark.global_min,
    )

    plot_regret(cfg, regrets)


if __name__ == "__main__":
    main()