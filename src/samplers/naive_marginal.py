from .base import OrthoBaseSampler
from src.utils import (
    build_diag_laplace_log_hyperposterior,
    sample_valid_thetas,
    OrthogonalLogEi
)
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf

class NaiveMarginalSampler(OrthoBaseSampler):
    def get_candidates(self, train_x, train_obj, train_con, bounds, pending_x):
        model = SingleTaskGP(train_x, train_obj)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        hyperposterior, param_spec, _ = build_diag_laplace_log_hyperposterior(
            fitted_model=model, hess_jitter=1e-4, max_log_std=0.10,
        )
        theta_samples, theta_models = sample_valid_thetas(
            model=model, hyperposterior=hyperposterior, param_specs=param_spec, S=self.mc_budget, train_x=train_x,
        )

        acqf = OrthogonalLogEi(
            model=model, best_f=train_obj.max(), hyperposterior=hyperposterior, 
            theta_samples=theta_samples, theta_models=theta_models,
            use_orthogonal_correction=False
        )

        candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
        return candidate