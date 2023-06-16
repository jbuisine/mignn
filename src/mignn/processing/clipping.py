from typing import List
import numpy as np
import torch

class ClipTransform():
    
    def __init__(self, max_clip: List[float], partial=True) -> None:
        self._max_clip = np.array(max_clip, dtype=float)
        self._partial = partial

    @property
    def enable_partial(self):
        return self._partial
    
    def partial_fit(self, _):
        # no need to fit
        return self
        
    def fit(self, _):
        # no need to fit
        return self
        
    def transform(self, X):
        
        if torch.is_tensor(X):
            X = X.numpy()
            
        X = np.clip(X, a_min=0, a_max=self._max_clip)
        return X
        
    def inverse_transform(self, X):
        
        if torch.is_tensor(X):
            X = X.numpy()
            
        # nothing to do...
        return X
  