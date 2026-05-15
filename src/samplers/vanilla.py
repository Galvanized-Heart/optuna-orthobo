import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseSampler


class VanillaBoTorchSampler(BaseSampler):
    def get_candidates(self, train_x, train_obj, train_con, bounds, pending_x):
        dim = train_x.shape[-1]

        # BoTorch handles normalization and standardization internally.
        model = SingleTaskGP(
            train_x,
            train_obj,
            input_transform=Normalize(d=dim, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_budget]))
        acqf = qLogExpectedImprovement(model, best_f=train_obj.max(), sampler=sampler)

        candidate, _ = optimize_acqf(
            acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=20
        )

        return candidate.clamp(bounds[0], bounds[1])