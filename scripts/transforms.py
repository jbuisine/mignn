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
        
        # perform scale of data
        data.x = torch.tensor(self._x_scaler.transform(data.x), dtype=torch.float)
        data.edge_attr = torch.tensor(self._edge_scaler.transform(data.edge_attr), dtype=torch.float)
        data.y = torch.tensor(self._y_scaler.transform(data.y.reshape(-1, 3)), dtype=torch.float)

        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
    
    
@functional_transform('signal_encoder')
class SignalEncoder(BaseTransform):

    def __init__(self, encoder_size=6):
        self._powers = torch.pow(2., torch.arange(encoder_size))
    
    def __call__(self, data: Data) -> Data:
        
        # perform encoder of data
        emb_data = []
    
        # for each data elements
        for c_data_x in data.x:
            
            out_data = torch.empty(0)
            
            for c_data in c_data_x:
                x_cos = torch.cos(c_data * self._powers)
                x_sin = torch.cos(c_data * self._powers)
                c_emb = torch.cat((c_data.unsqueeze(0), x_cos, x_sin), 0)
                out_data = torch.cat((out_data, c_emb), 0)
                
            emb_data.append(out_data)
        
        data.x = torch.stack(emb_data)

        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'