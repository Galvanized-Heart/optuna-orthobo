from .base import OrthoBaseSampler

class OrthoBoSampler(OrthoBaseSampler):
    def get_candidates(self, train_x, train_obj, train_con, bounds, pending_x):
        # TODO: Implement orthobo in optuna
        pass