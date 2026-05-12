import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from .base import OrthoBaseSampler

class VanillaBoTorchSampler(OrthoBaseSampler):
    def get_candidates(self, train_x, train_obj, train_con, bounds, pending_x):
        # 1. Fit the GP
        model = SingleTaskGP(train_x, train_obj)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        # 2. Setup the Sampler (Sobol by default for Vanilla)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_budget]))
        
        # 3. Optimize Acquisition
        acqf = qLogExpectedImprovement(model, best_f=train_obj.max(), sampler=sampler)
        candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
        
        return candidate