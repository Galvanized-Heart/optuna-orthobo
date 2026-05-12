import torch
from dataclasses import dataclass
from typing import List, Tuple
from torch import Tensor

@dataclass
class ThetaField:
    path: str
    shape: torch.Size
    size: int
    transform: str = "log"

@dataclass
class ThetaSpec:
    fields: List[ThetaField]
    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.fields)

def _get_attr_by_path(obj, path: str):
    out = obj
    for part in path.split("."):
        out = getattr(out, part)
    return out

def _set_attr_by_path(obj, path: str, value):
    parts = path.split(".")
    parent = obj
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)

SUPPORTED_POSITIVE_PATHS = [
    "likelihood.noise",
    "covar_module.outputscale",
    "covar_module.variance",
    "covar_module.lengthscale",
]

def discover_theta_spec(model) -> ThetaSpec:
    fields = []
    for path in SUPPORTED_POSITIVE_PATHS:
        try:
            val = _get_attr_by_path(model, path)
            if torch.is_tensor(val):
                fields.append(ThetaField(path=path, shape=val.shape, size=val.numel(), transform="log"))
        except AttributeError:
            continue
    if not fields:
        raise RuntimeError("No supported hyperparameters found.")
    return ThetaSpec(fields=fields)

def extract_log_theta(model) -> Tuple[Tensor, ThetaSpec]:
    spec = discover_theta_spec(model)
    pieces = []
    for field in spec.fields:
        val = _get_attr_by_path(model, field.path).detach().reshape(-1).clamp_min(1e-12)
        pieces.append(val.log())
    return torch.cat(pieces), spec

def unpack_log_theta(u: Tensor, spec: ThetaSpec) -> dict:
    out = {}
    offset = 0
    for field in spec.fields:
        chunk = u[offset : offset + field.size].view(field.shape)
        out[field.path] = chunk
        offset += field.size
    return out

def set_theta_from_log_(model, u: Tensor, spec: ThetaSpec) -> None:
    path_to_log = unpack_log_theta(u, spec)
    with torch.no_grad():
        for field in spec.fields:
            _set_attr_by_path(model, field.path, path_to_log[field.path].exp())
    model.prediction_strategy = None
