from typing import List
from abc import ABC
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from mignn.dataset import PathLightDataset

class MaskedScaler(ABC):
    
    def __init__(self, scaler, mask: List[bool], partial=True) -> None:
        self._scaler = scaler
        self._mask = np.array(mask, dtype=bool)
        self._partial = partial

    @property
    def enable_partial(self):
        return self._partial
    
    def partial_fit(self, x):
        self._scaler.partial_fit(x[:, self._mask])
        return self
        
    def fit(self, X):
        self._scaler.fit(X[:, self._mask])
        return self
        
    def transform(self, X):
        
        if torch.is_tensor(X):
            X = X.numpy()
            
        X[:, self._mask] = self._scaler.transform(X[:, self._mask])
        return X
        
    def inverse_transform(self, X):
        
        if torch.is_tensor(X):
            X = X.numpy()
            
        # create new axis if necessary
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
            
        X[:, self._mask] = self._scaler.inverse_transform(X[:, self._mask])
        return X
    
class MaskedMinMaxScaler(MaskedScaler):
    
    def __init__(self, mask: List[bool]) -> None:
        super().__init__(MinMaxScaler(), mask)
        
class MaskedStandardScaler(MaskedScaler):
    
    def __init__(self, mask: List[bool]) -> None:
        super().__init__(StandardScaler(), mask)
        
class MaskedRobustScaler(MaskedScaler):
    
    def __init__(self, mask: List[bool]) -> None:
        super().__init__(RobustScaler(), mask, False)
        
    
class MaskedLogScaler(MaskedScaler):
    
    def __init__(self, mask: List[bool], partial=True) -> None:
        # no scaler required
        super().__init__(None, mask, partial)
    
    def partial_fit(self, x):
        # no need to fit
        return self
        
    def fit(self, X):
        # no need to fit
        return self
        
    def transform(self, X):
        
        if torch.is_tensor(X):
            X = X.detach().numpy()
                
        X_copy = X.copy()
        
        # check shape
        if len(X_copy.shape) == 1:
            X_copy = X_copy[np.newaxis, :]
        
        X_copy[:, self._mask] = np.log(X_copy[:, self._mask] + 1)
        
        return X_copy
        
    def inverse_transform(self, X):
        
        if torch.is_tensor(X):
            X = X.detach().numpy()
            
        X_copy = X.copy()
        
        # check shape
        if len(X_copy.shape) == 1:
            X_copy = X_copy[np.newaxis, :]
            
        # rescaled only expected features
        X_copy[:, self._mask] = np.exp(X_copy[:, self._mask]) - 1
        return X_copy
    

class ScalersManager():
    """Manage scaling preprocessing of graph data
    """
    
    _dataset_field_names = ['x', 'edge_attr', 'y_direct', 'y_indirect', 'y_total', 'origin', 'direction']
    _expected_keys = ['x_node', 'x_edge', 'y_direct', 'y_indirect', 'y_total', 'origin', 'direction']
    
    def __init__(self, config: dict) -> None:
        """Config scaler order by field is preserved
        """
        
        # check config first
        if not all([ k in self._expected_keys for k in config.keys() ]):
            raise AttributeError(f'Provided configuration expected fields: {self._expected_keys}')
        
        self._normalizers = {}
        for field, models in config.items():
            
            self._normalizers[field] = []
            
            # init each masked normalizer for current field
            for name, mask in models.items():
                normalizer = self.__init_normalizer(name, mask)
                
                if normalizer is not None:
                    self._normalizers[field].append(normalizer)
    
    def __init_normalizer(self, normalizer_name: str, mask: List[bool]):
        """Get the expected Masked normalizer
        """
        
        if normalizer_name == 'MinMax':
            return MaskedMinMaxScaler(mask)

        if normalizer_name == 'Robust':
            return MaskedRobustScaler(mask)
        
        if normalizer_name == 'Standard':
            return MaskedStandardScaler(mask)
        
        if normalizer_name == 'Log':
            return MaskedLogScaler(mask)
        
        return None
    
    
    def fit(self, dataset: PathLightDataset):
            
        for k_i, key in enumerate(self._expected_keys):
            
            if key in self._normalizers:
                    
                field_key = self._dataset_field_names[k_i]
                # transform each data
                data = dataset.data[field_key]
                
                if field_key in ['y_direct', 'y_indirect', 'y_total', 'origin', 'direction']:
                    data = data.reshape((dataset.len(), -1))    
                    
                for norm_model in self._normalizers[key]:
                    data = norm_model.fit_transform(data)
                    
                    
    def partial_fit(self, dataset: PathLightDataset):
        
        for k_i, key in enumerate(self._expected_keys):
            
            if key in self._normalizers:
                    
                field_key = self._dataset_field_names[k_i]
                # transform each data
                data = dataset.data[field_key]
                
                if field_key in ['y_direct', 'y_indirect', 'y_total', 'origin', 'direction']:
                    # number of predictions in dataset
                    data = data.reshape((dataset.len(), -1))
                
                for norm_model in self._normalizers[key]:
                    norm_model.partial_fit(data)
                    data = norm_model.transform(data)
            
        
    def inverse_transform_field(self, field_name: str, data):
        
        c_data = data.copy()
        
        if field_name in self._normalizers:
                    
            # apply transformations in reverse order
            for norm_model in self._normalizers[field_name][::-1]:
                c_data = norm_model.inverse_transform(c_data)
        
        return c_data
    
    def get_scalers_from_field(self, field_name: str):
        
        if field_name in self._normalizers:
            return self._normalizers[field_name]
        
        return None
    
    def get_scalers_fields(self):
        
        return self._normalizers.keys()
    
    @property
    def enable_partial(self):
        
        # check if at least one model require fit on all data
        for _, normalizers in self._normalizers.items():
            
            for norm_model in normalizers:
                if not norm_model.enable_partial:
                    return False
        
        return True