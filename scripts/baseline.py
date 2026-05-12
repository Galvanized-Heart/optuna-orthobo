import optuna
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.test_functions import Hartmann
from optuna_integration import BoTorchSampler

# Init hartmann function
dim = 6
test_function = Hartmann(dim=dim)

# Hartmann-6 global minimum = -3.32237 (https://www.sfu.ca/~ssurjano/hart6.html)
GLOBAL_MINIMUM = -3.32237 

def objective(trial):
    # Suggest parameters between 0.0 and 1.0 (Hartmann bounds)
    x_values = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)]
    
    X_tensor = torch.tensor([x_values], dtype=torch.float64)
    
    # Evaluate function
    result = test_function(X_tensor).item()
    return result

def run_baseline_test():
    print("Starting Baseline Optuna/BoTorch Test...")
    
    # Use 10 startup trials before BO
    sampler = BoTorchSampler(n_startup_trials=10)
    
    # Set study to minimize
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    n_trials = 50
    study.optimize(objective, n_trials=n_trials)
    
    best_so_far = []
    current_best = float('inf')
    
    for t in study.trials:
        if t.value is not None and t.value < current_best:
            current_best = t.value
        regret = current_best - GLOBAL_MINIMUM
        best_so_far.append(regret)
        
    print(f"Final Regret: {best_so_far[-1]:.6f}")
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(range(n_trials), best_so_far, label="Default BoTorchSampler", color='blue', linewidth=2)
    plt.axvline(x=10, color='gray', linestyle='--', label='End of Random Startup')
    
    plt.yscale('log') # Log scale is standard for regret plots
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far Regret (Log Scale)")
    plt.title("Phase 1: Hartmann-6 Baseline Verification")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/phase_1_baseline.png")
    print("Plot saved to figures/phase_1_baseline.png")

if __name__ == "__main__":
    run_baseline_test()