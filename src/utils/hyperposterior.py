import copy
import torch
from dataclasses import dataclass
from torch import Tensor
from torch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from .theta_extraction import *

@dataclass
class GaussianHyperPosterior:
    mean: Tensor
    cov: Tensor
    def sample(self, S: int) -> Tensor:
        return MultivariateNormal(self.mean, covariance_matrix=self.cov).rsample((S,))
    def score(self, u: Tensor) -> Tensor:
        diff = u - self.mean
        return -torch.linalg.solve(self.cov, diff.T).T

def neg_log_post_at_log_theta(u: Tensor, model_template, spec: ThetaSpec) -> Tensor:
    model = copy.deepcopy(model_template)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    model.likelihood.train()
    set_theta_from_log_(model, u, spec)
    output = model(model.train_inputs[0])
    return -mll(output, model.train_targets)

def build_diag_laplace_log_hyperposterior(fitted_model, hess_jitter: float = 1e-4, max_log_std: float = 0.10):
    u_hat, spec = extract_log_theta(fitted_model)
    u_hat = u_hat.detach().clone().requires_grad_(True)
    def objective(u_vec: Tensor) -> Tensor:
        return neg_log_post_at_log_theta(u=u_vec, model_template=fitted_model, spec=spec)
    
    H = torch.autograd.functional.hessian(objective, u_hat).detach()
    H = 0.5 * (H + H.T)
    H_diag = torch.diag(H).clamp_min(hess_jitter)
    
    std = torch.clamp(1.0 / torch.sqrt(H_diag), max=max_log_std)
    hyperposterior = GaussianHyperPosterior(mean=u_hat.detach(), cov=torch.diag(std.pow(2)).detach())
    return hyperposterior, spec, u_hat.detach()

def sample_valid_thetas(model, hyperposterior, param_specs, S, train_x, max_tries=1000):
    accepted_models, accepted_thetas = [], []
    thetas = hyperposterior.sample(2 * S)
    idx, tries = 0, 0
    while len(accepted_models) < S and tries < max_tries:
        theta = thetas[idx]
        tries += 1
        try:
            model_s = copy.deepcopy(model)
            model_s.prediction_strategy = None
            set_theta_from_log_(model=model_s, u=theta, spec=param_specs)
            _ = model_s.posterior(train_x[:1]) # Test if model is valid
            accepted_models.append(model_s)
            accepted_thetas.append(theta)
        except Exception:
            pass
        idx += 1
    if len(accepted_models) < S:
        raise RuntimeError("Could not obtain enough valid theta samples.")
    return torch.stack(accepted_thetas, dim=0), accepted_models
