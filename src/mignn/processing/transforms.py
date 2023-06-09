import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform

from .scalers import ScalersManager

@functional_transform('scaler_transform')
class ScalerTransform(BaseTransform):

    def __init__(self, scalers_manager: ScalersManager):
        
        # store scaler fields
        self._scalers = scalers_manager
        self._fields = scalers_manager.get_scalers_fields()
            
    def __call__(self, data: Data) -> Data:
        
        # perform scale of data (if scalers exists)
        if 'x_node' in self._fields:
            x_scalers = self._scalers.get_scalers_from_field('x_node')
            for x_scaler in x_scalers:
                data.x = torch.tensor(x_scaler.transform(data.x), dtype=torch.float)
        
        # edge attributes
        if 'x_edge' in self._fields:
            
            edge_scalers = self._scalers.get_scalers_from_field('x_edge')
            # only if graph has connections
            # TODO: check when scaler is a Encoding one (empty ray may cause error)
            if data.edge_attr.size()[0] > 0: 
                for edge_scaler in edge_scalers:
                    data.edge_attr = torch.tensor(edge_scaler.transform(data.edge_attr), dtype=torch.float)
        
        # targets and camera attributes
        for c_key in ['y_direct', 'y_indirect', 'y_total', 'origin', 'direction']:
            if c_key in self._fields:
            
                # reshape data
                if len(data[c_key].shape) == 1:
                    data[c_key] = data[c_key].unsqueeze(0)
                    
                c_scalers = self._scalers.get_scalers_from_field(c_key)
                
                for c_scaler in c_scalers:
                    data[c_key] = torch.tensor(c_scaler.transform(data[c_key]), dtype=torch.float)
                   
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
    
    
@functional_transform('signal_encoder')
class SignalEncoder(BaseTransform):

    def __init__(self, encoder_size=6, mask= None, log_space=False):
        
        # TODO: specify which data need to be transformed (check other transform)
        self.n_freqs = encoder_size
        self.log_space = log_space
        self.default = lambda x: x # keep default value of feature
        self.embed_fns = []
        self.mask = {k: torch.tensor(c_mask, dtype=torch.uint8) for k, c_mask in mask.items()} 

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
            
    def __apply(self, x, mask_key):
        
        # apply transformation on mask if required
        xx = x
        if self.mask[mask_key] is not None:
            if len(self.mask[mask_key]) != len(x):
                raise ValueError(f'Invalid mask size for {mask_key}. Mask size must be {len(x)}')
            xx = x[self.mask[mask_key]]
        
        if self.mask[mask_key] is None or len(xx) == 0:
            # if no mask, then apply nothing, just keep the previous field (same as mask of full zeros)
            return torch.empty(0, dtype=torch.float32)
            
        return torch.concat([fn(xx) for fn in self.embed_fns], dim=-1)
    
    def __call__(self, data: Data) -> Data:
        
        # transform if field using provided mask
        data.x = torch.stack([ torch.cat([self.default(x), self.__apply(x, 'x_node') ]) for x in data.x])
        
        # check if there is edges data
        if data.edge_attr.size()[0] > 0:
            data.edge_attr = torch.stack([ torch.cat([self.default(e), self.__apply(e, 'x_edge') ]) for e in data.edge_attr])
            
        # target radiance attributes
        data.y_total = torch.stack([ torch.cat([self.default(y), self.__apply(y, 'y_total') ]) for y in data.y_total])
        data.y_direct = torch.stack([ torch.cat([self.default(y), self.__apply(y, 'y_direct') ]) for y in data.y_direct])
        data.y_indirect = torch.stack([ torch.cat([self.default(y), self.__apply(y, 'y_indirect') ]) for y in data.y_indirect])

        # camera attributes
        data.origin = torch.stack([ torch.cat([self.default(o), self.__apply(o, 'origin') ]) for o in data.origin])
        data.direction = torch.stack([ torch.cat([self.default(d), self.__apply(d, 'direction') ]) for d in data.direction])
        
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'