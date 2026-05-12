from optuna_integration import BoTorchSampler

class BaseSampler(BoTorchSampler):
    """Base class to standardize custom sampling logic."""
 
    def __init__(
        self,
        n_startup_trials: int = 10,
        mc_budget: int = 64,
        sampler_name: str = "BaseSampler",
    ):
        super().__init__(candidates_func=self.get_candidates, n_startup_trials=n_startup_trials)
        self.mc_budget = mc_budget
        self.sampler_name = sampler_name
 
    def get_candidates(self, train_x, train_obj, train_con, bounds, pending_x):
        raise NotImplementedError("Subclasses must implement get_candidates.")
