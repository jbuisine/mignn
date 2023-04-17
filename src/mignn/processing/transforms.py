import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform

@functional_transform('scaler_transform')
class ScalerTransform(BaseTransform):

    def __init__(self, scalers):
        self._x_scaler = scalers['x_node']
        self._edge_scaler = scalers['x_edge']
        self._y_scaler = scalers['y']

    def __call__(self, data: Data) -> Data:
        
        # perform scale of data (if scaler exists)
        if self._x_scaler is not None:
            data.x = torch.tensor(self._x_scaler.transform(data.x), dtype=torch.float)
        
        if self._edge_scaler is not None:
            data.edge_attr = torch.tensor(self._edge_scaler.transform(data.edge_attr), dtype=torch.float)
        
        if self._y_scaler is not None:
            data.y = torch.tensor(self._y_scaler.transform(data.y.reshape(-1, 3)), dtype=torch.float)

        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
    
    
@functional_transform('signal_encoder')
class SignalEncoder(BaseTransform):

    def __init__(self, encoder_size=6, log_space=False):
        
        # TODO: specify which data need to be transformed (check other transform)
        self.n_freqs = encoder_size
        self.log_space = log_space
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
            
    def __apply(self, x):
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
    
    def __call__(self, data: Data) -> Data:
        
        data.x = torch.stack([self.__apply(x) for x in data.x])

        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'