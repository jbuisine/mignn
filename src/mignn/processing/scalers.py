from abc import ABC
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class LogScaler(ABC):
    
    def __init__(self, scaler) -> None:
        self._scaler = scaler
    
    def partial_fit(self, x):
        x_log_based = np.log(x + 1)
        self._scaler.partial_fit(x_log_based)
        return self
        
    def fit(self, X):
        X_log_based = np.log(X + 1)
        self._scaler.fit(X_log_based)
        return self
        
    def transform(self, X):
        X_log_based = np.log(X + 1)
        return self._scaler.transform(X_log_based)
        
    def inverse_transform(self, X):
        X_transformed = self._scaler.inverse_transform(X)
        return np.exp(X_transformed) - 1
    
    
class LogMinMaxScaler(LogScaler):
    
    def __init__(self) -> None:
        super().__init__(MinMaxScaler())
        
class LogStandardScaler(LogScaler):
    
    def __init__(self) -> None:
        super().__init__(StandardScaler())
        
class LogRobustScaler(LogScaler):
    
    def __init__(self) -> None:
        super().__init__(RobustScaler())
        
    def partial_fit(self, x):
        return None