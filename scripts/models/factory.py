from .gcn_model import GNNL, GNNL_VP, GNNL_VPP
from .nerf import SimpleNeRf
from .param import ModelParam

class ModelFactory():
    
    @staticmethod
    def from_params(model: ModelParam):
        
        if model.kind == 'nerf':
            return ModelFactory.__create_nerf(model.name, model.params)
        
        if model.kind == 'gnn':
            return ModelFactory.__create_gnn(model.name, model.params)
        
        return None
    
    
    @staticmethod
    def check_fields(expected, current_fields):
        
        for field in expected:
            if field not in current_fields:
                raise ValueError(f'{field} parameter in order to instantiate model')
    
    
    @staticmethod
    def __create_nerf(name:str, params: dict):
        
        common_expected_fields = [
            'n_features', 
            'hidden_size', 
            'n_hidden_layers'
        ]
        
        if name == "simple":
            
            ModelFactory.check_fields(common_expected_fields, params)
            
            # construct expected model
            return SimpleNeRf(n_features=params['n_features'], 
                    hidden_size=params['hidden_size'],
                    n_hidden_layers=['n_dense_layers'])
            
        return None
            
        
    @staticmethod
    def __create_gnn(name:str, params: dict):
        
        common_expected_fields = [
            'graph_hidden_channels', 
            'dense_hidden_layers',
            'n_dense_layers',
            'latent_space',
            'n_features'
        ]
        
        if name == 'simple':
            
            ModelFactory.check_fields(common_expected_fields, params)
            
            # construct expected model
            return GNNL(graph_hidden_channels=params['graph_hidden_channels'], 
                    dense_hidden_layers=params['dense_hidden_layers'],
                    n_dense_layers=params['n_dense_layers'],
                    latent_size=params['latent_space'],
                    n_features=params['n_features'])
            
        if name == 'simple_camera':
            
            ModelFactory.check_fields(common_expected_fields + ['n_camera_features'], params)
            
            return GNNL_VP(graph_hidden_layers=params['graph_hidden_channels'], 
                    dense_hidden_layers=params['dense_hidden_layers'],
                    n_dense_layers=params['n_dense_layers'],
                    latent_size=params['latent_space'],
                    n_features=params['n_features'],
                    n_camera_features=params['n_camera_features'])
            
        if name == 'simple_viewpoint':
            
            ModelFactory.check_fields(common_expected_fields + ['n_camera_features'], params)
            
            return GNNL_VPP(graph_hidden_layers=params['graph_hidden_channels'], 
                    dense_hidden_layers=params['dense_hidden_layers'],
                    n_dense_layers=params['n_dense_layers'],
                    latent_size=params['latent_space'],
                    n_features=params['n_features'],
                    n_camera_features=params['n_camera_features'])
            
        return None