import torch
from torch import Tensor
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.acquisition import AcquisitionFunction

class OrthogonalLogEi(AcquisitionFunction):
    def __init__(self, model, best_f, hyperposterior, theta_samples, theta_models, eps: float = 1e-12, cov_jitter: float = 1e-6, use_orthogonal_correction: bool = True):
        super().__init__(model=model)
        self.eps = eps
        self.cov_jitter = cov_jitter
        self.use_orthogonal_correction = use_orthogonal_correction
        self.g_cache = hyperposterior.score(theta_samples)
        self.g_centered_cache = self.g_cache - self.g_cache.mean(dim=0, keepdim=True)
        self.g_cov_cache = torch.cov(self.g_cache.T)
        self.theta_acqfns = [LogExpectedImprovement(model=m, best_f=best_f) for m in theta_models]

    def forward(self, X: Tensor) -> Tensor:
        # Evaluate standard EI for all S models
        vals2 = [torch.exp(fn(X)) for fn in self.theta_acqfns]
        h = torch.stack(vals2, dim=0) # [S, batch]

        if not self.use_orthogonal_correction:
            return torch.log(torch.clamp_min(h.mean(dim=0), self.eps))

        g = self.g_cache
        g_centered = self.g_centered_cache
        cov_gg = self.g_cov_cache
        S = g.shape[0]

        h_centered = h - h.mean(dim=0, keepdim=True)

        # Apply Orthogonal Control Variate Matrix Math
        cov_gg = cov_gg + self.cov_jitter * torch.eye(cov_gg.shape[0], dtype=cov_gg.dtype, device=cov_gg.device)
        cov_gh = (g_centered.T @ h_centered) / max(S - 1, 1)

        gamma = torch.linalg.solve(cov_gg, cov_gh)
        orth = h - (g @ gamma)
        
        return torch.log(torch.clamp_min(orth.mean(dim=0), self.eps))