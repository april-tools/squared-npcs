import numpy as np
from torch import inf
from torch.optim import lr_scheduler


class ReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    def is_better(self, a, best):
        # Fix relative improvement implementation to work with any loss in R, not just positive losses
        if self.threshold_mode == 'rel':
            if self.mode == 'min':
                if best == inf:
                    return True
                if np.abs(best) < 1:
                    return a < best - self.threshold
                return a < best - self.threshold * np.abs(best)
            else:
                if best == -inf:
                    return True
                if np.abs(best) < 1:
                    return a > best + self.threshold
                return a > best + self.threshold * np.abs(best)
        return super().is_better(a, best)
