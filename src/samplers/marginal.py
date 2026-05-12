from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseSampler
from src.utils import (
    build_diag_laplace_log_hyperposterior,
    sample_valid_thetas,
    OrthogonalLogEi,
)


class MarginalBoTorchSampler(BaseSampler):
    """
    Unified Marginal BO sampler.

    When use_orthobo=True (default), applies the orthogonal control variate
    correction from the OrthoBO paper, yielding variance-reduced acquisition
    estimates. When use_orthobo=False, falls back to naive MC marginalisation.

    Input normalization and output standardization are handled internally by
    BoTorch's Normalize and Standardize transforms — optimize_acqf operates
    in the original parameter space throughout.
    """

    def __init__(
        self,
        n_startup_trials: int = 10,
        mc_budget: int = 64,
        use_orthobo: bool = True,
        sampler_name: str = "MarginalBO",
    ):
        super().__init__(
            n_startup_trials=n_startup_trials,
            mc_budget=mc_budget,
            sampler_name=sampler_name,
        )
        self.use_orthobo = use_orthobo

    def get_candidates(self, train_x, train_obj, train_con, bounds, pending_x):
        dim = train_x.shape[-1]

        # BoTorch transforms handle normalization internally
        model = SingleTaskGP(
            train_x,
            train_obj,
            input_transform=Normalize(d=dim, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Build Laplace approximation of the log-hyperparameter posterior
        hyperposterior, param_spec, _ = build_diag_laplace_log_hyperposterior(
            fitted_model=model,
            hess_jitter=1e-4,
            max_log_std=0.10,
        )

        # Draw S valid GP models from the hyperposterior
        theta_samples, theta_models = sample_valid_thetas(
            model=model,
            hyperposterior=hyperposterior,
            param_specs=param_spec,
            S=self.mc_budget,
            train_x=train_x,
        )

        # Orthogonal (or naive) acquisition
        acqf = OrthogonalLogEi(
            model=model,
            best_f=train_obj.max(),
            hyperposterior=hyperposterior,
            theta_samples=theta_samples,
            theta_models=theta_models,
            use_orthogonal_correction=self.use_orthobo,
        )

        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        return candidate