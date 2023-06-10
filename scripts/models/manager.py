import os
from abc import abstractmethod

from typing import Iterable
from typing import List

import torch
from torch.autograd import Variable
from torchmetrics import R2Score

from .factory import ModelFactory
from .param import ModelParam
from .gcn_model import GNNL_VPP

class ModelManager():
    
    EXPECTED_MODELS = ['nerf', 'gnn']
    
    def __init__(self, params: list[ModelParam]) -> None:
        """
        Args:
            nerf_name: str
            params (dict): must contain for each model its expected params
        """
        self._models = {}
        self._optimizers = {}
        self._losses = {}
        
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._r2_metric = R2Score().to(self._device)
        self._batchs = {'train': 0, 'test': 0 }
        
        # check models
        self._check_models(params)
        self._params = params
        
        self._build_models()
        self._init_metrics()
    
    def _check_models(self, model_params: List[ModelParam]):
        
        for model_param in model_params:
            if model_param.kind not in ModelManager.EXPECTED_MODELS:
                raise ValueError(f'{model_param.kind} model not expected')
    

    def _init_loss(self, loss_name):
        """Get the expected torch loss
        """
        
        if loss_name == 'MSE':
            return torch.nn.MSELoss()

        if loss_name == 'MAE':
            return torch.nn.L1Loss()
        
        if loss_name == 'Huber':
            return torch.nn.HuberLoss()
        
        return None
    
    def _init_optimizer(self, model, optimizer_name):
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=0.001)
        
        return None
    
    def __get_all_metrics_values(self, d):
        
        if isinstance(d, dict):
            for v in d.values():
                yield from self.__get_all_metrics_values(v)
        elif isinstance(d, Iterable) and not isinstance(d, str): # or list, set, ... only
            for v in d:
                yield from self.__get_all_metrics_values(v)
        else:
            yield d 
            
    def __get_all_metrics_keys(self, d):
        
        if isinstance(d, dict):
            for v in d.keys():
                yield from self.__get_all_metrics_keys(v)
        elif isinstance(d, Iterable) and not isinstance(d, str): # or list, set, ... only
            for v in d:
                yield from self.__get_all_metrics_keys(v)
        else:
            yield d 
            
    def metrics_header(self):
        return self.__get_all_metrics_keys(self._metrics)
        
    def metrics_values(self):
        return self.__get_all_metrics_values(self._metrics)
    
    def save(self, folder):
        
        os.makedirs(folder, exist_ok=True)
        
        for kind, model in self._models.items():
            torch.save(model.state_dict(), f'{folder}/model_{kind}.pt')
            
        for kind, optimizer in self._optimizers.items():
            torch.save(optimizer.state_dict(), f'{folder}/optimizer_{kind}.pt')
            
    def load(self, folder):
        
        for kind, model in self._models.items():
            model.load_state_dict(torch.load( f'{folder}/model_{kind}.pt'))
            
        for kind, optimizer in self._optimizers.items():
            optimizer.load_state_dict(torch.load( f'{folder}/optimizer_{kind}.pt'))
    
    def train(self):
        for _, model in self._models.items():
            model.train()
            
    def eval(self):
        for _, model in self._models.items():
            model.eval()
        
    def _init_metrics(self):
        
        self._batchs = {'train': 0, 'test': 0 }
        
        self._metrics = {
            'train': {},
            'test': {}
        }
        
        for model in self._params:
            if model.kind in self._models:
                self._metrics['train'][model.kind] = {
                    'loss': 0,
                    'r2': 0
                }
                
                self._metrics['test'][model.kind] = {
                    'loss': 0,
                    'r2': 0
                }
                
    @abstractmethod
    def _build_models(self):
        pass
        
    @abstractmethod
    def test(self, data) -> dict:
        """Perform test on specific data and return obtained results
        """
        self._batchs['test'] += 1
        
    @abstractmethod
    def step(self, data) -> dict:
        """Perform one train step from batch data and return 
        """
        self._batchs['train'] += 1
    
    @abstractmethod
    def predict(self, data, scalers):
        pass

    @abstractmethod
    def score(self, mode, metric='r2'):
        pass

        
    def clear_metrics(self):
        self._init_metrics()
        
    def information(self, mode):

        if mode in self._metrics:
            
            c_metrics = self._metrics[mode]
            
            model_str = []
            for key, values in c_metrics.items():
                # divide per number of batchs
                model_str.append(f'[{key}] {", ".join([ f"{k}: {v / self._batchs[mode]:.5f}" for k, v in values.items()])}')
                
            return ' - '.join(model_str)
        
        return None
            
    @property
    def metrics(self):
        return self._metrics
        
    @property
    def models(self):
        return self._models

    @property
    def optimizers(self):
        return self._optimizers
        
class SimpleModelManager(ModelManager):
    
    def __init__(self, params: list[ModelParam]) -> None:
        super().__init__(params)
            
    def _build_models(self):
        
        # only instantiate models with gnn kind
        for model_params in self._params:
            if model_params.kind == 'gnn':
                
                self._models['gnn'] = ModelFactory.from_params(model_params).to(self._device)
                # default optimizer
                self._optimizers['gnn'] = self._init_optimizer(self._models['gnn'], 'adam')
                self._losses['gnn'] = self._init_loss(model_params.loss).to(self._device) 
      
    def test(self, data) -> dict:
        super().test(data)
        
        y_predicted = self._models['gnn'](data)
        
        # predict whole radiance
        loss = self._models['gnn'].metric(data, y_predicted, data.y_total, self._losses['gnn'])
        self._metrics['test']['gnn']['loss'] += loss.item()
        r2_loss = self._models['gnn'].metric(data, y_predicted, data.y_total, self._r2_metric)
        self._metrics['test']['gnn']['r2'] += r2_loss.item()

                    
    def step(self, data) -> dict:
        super().step(data)
        
        # TRAIN GNN
        self._optimizers['gnn'].zero_grad() 
        
        y_predicted = self._models['gnn'](data)

        loss = self._models['gnn'].metric(data, y_predicted, data.y_total, self._losses['gnn'])
        self._metrics['train']['gnn']['loss'] += loss.item()
        
        loss.backward()
        self._optimizers['gnn'].step()
        
        r2_loss = self._models['gnn'].metric(data, y_predicted, data.y_total, self._r2_metric)
        self._metrics['train']['gnn']['r2'] += r2_loss.item()
        
        
    def predict(self, data, scalers) -> List[torch.Tensor]:
        
        y_predicted = self._models['gnn'](data).detach().cpu().numpy()
        
        # predict whole radiance
        y_radiance = self._models['gnn'].radiance_from_predictions(y_predicted)
        
        # Radiance (indirect and direct) must be the 3 thirds features to predict
        y_total_target = data.y_total.detach().cpu().numpy()
        
        # no need to rescale input radiance
        input_radiance = data.direct_radiance.detach().cpu().numpy() + data.indirect_radiance.detach().cpu().numpy()
            
        # rescaled if necessary
        if scalers.get_scalers_from_field('y_total') is not None:
            y_radiance = scalers.inverse_transform_field('y_total', y_radiance)
            y_total_target = scalers.inverse_transform_field('y_total', y_total_target)
                    
        return input_radiance, y_radiance, y_total_target


    def score(self, mode, metric='r2'):
        
        return self._metrics[mode]['gnn'][metric] / self._batchs[mode]
        
class SeparatedModelManager(ModelManager):
    
    def __init__(self, params: list[ModelParam]) -> None:
        super().__init__(params)
            
        self._global_criterion = torch.nn.MSELoss().to(self._device)
            
    def _init_metrics(self):
        super()._init_metrics()
        
        # add global criterion
        for mode in ['test', 'train']:
            self._metrics[mode]['global'] = {
                'loss': 0,
                'r2': 0
            }
                
        
    def _build_models(self):
                
        # instantiate all models with specific kind
        for model_params in self._params:
                
            kind = model_params.kind
        
            self._models[kind] = ModelFactory.from_params(model_params).to(self._device)
            self._optimizers[kind] = self._init_optimizer(self._models[kind], 'adam')
            self._losses[kind] = self._init_loss(model_params.loss).to(self._device)
        
    def test(self, data) -> dict:
        super().test(data)
        
        camera_features = torch.cat([data.origin, data.direction], dim=1).to(self._device)
        
        # Test using NeRF
        o_direct_radiance = self._models['nerf'](camera_features)
        # o_direct_radiance = o_direct_radiance[:, :-1] * o_direct_radiance[:, -1]
        loss = self._losses['nerf'](o_direct_radiance.flatten(), data.y_direct.flatten())  # Compute the loss.
        
        self._metrics['test']['nerf']['loss'] += loss.item()
        self._metrics['test']['nerf']['r2'] += self._r2_metric(o_direct_radiance.flatten(), data.y_direct.flatten()).item()
        
        # Test using GNN
        o_predicted = self._models['gnn'](data)
        # retrieve expected radiance
        o_indirect_radiance = self._models['gnn'].radiance_from_predictions(o_predicted)

        loss = self._models['gnn'].metric(data, o_predicted, data.y_indirect, self._losses['gnn'])
        self._metrics['test']['gnn']['loss'] += loss.item()
        r2_loss = self._models['gnn'].metric(data, o_predicted, data.y_indirect, self._r2_metric)
        self._metrics['test']['gnn']['r2'] += r2_loss.item()
        
        expected_radiance = Variable(data.y_direct.flatten() + data.y_indirect.flatten(), requires_grad=True)
        o_radiance = Variable(o_direct_radiance.flatten() + o_indirect_radiance.flatten(), requires_grad=True)
        
        # Propagate NERF global error into the two networks
        loss = self._global_criterion(o_radiance, expected_radiance)
        
        self._metrics['test']['global']['loss'] += loss.item()
        self._metrics['test']['global']['r2'] += self._r2_metric(o_radiance, expected_radiance).item()
        
                    
    def step(self, data) -> dict:
        super().step(data)
        
        # TRAIN NERF    
        self._optimizers['nerf'].zero_grad()
        
        camera_features = torch.cat([data.origin, data.direction], dim=1)
        
        o_direct_radiance = self._models['nerf'](camera_features)
        # o_direct_radiance = o_direct_radiance[:, :-1] * o_direct_radiance[:, -1]
        loss = self._losses['nerf'](o_direct_radiance.flatten(), data.y_direct.flatten())  # Compute the loss.
        
        self._metrics['train']['nerf']['loss'] += loss.item()
        self._metrics['train']['nerf']['r2'] += self._r2_metric(o_direct_radiance.flatten(), data.y_direct.flatten()).item()
        
        loss.backward()
        self._optimizers['nerf'].step()  
        
        # TRAIN GNN
        self._optimizers['gnn'].zero_grad() 
        
        # Specific case: predict also for each graph the camera and direction position    
        o_predicted = self._models['gnn'](data)
        
        # retrieve expected radiance
        o_indirect_radiance = self._models['gnn'].radiance_from_predictions(o_predicted)

        loss = self._models['gnn'].metric(data, o_predicted, data.y_indirect, self._losses['gnn'])
        self._metrics['train']['gnn']['loss'] += loss.item()
        r2_loss = self._models['gnn'].metric(data, o_predicted, data.y_indirect, self._r2_metric)
        self._metrics['train']['gnn']['r2'] += r2_loss.item()
        
        loss.backward()
        self._optimizers['gnn'].step()  
        
        expected_radiance = Variable(data.y_direct.flatten() + data.y_indirect.flatten(), requires_grad=True)
        o_radiance = Variable(o_direct_radiance.flatten() + o_indirect_radiance.flatten(), requires_grad=True)
        
        # Propagate NERF global error into the two networks
        self._optimizers['gnn'].zero_grad() 
        self._optimizers['nerf'].zero_grad() 
        
        loss = self._global_criterion(o_radiance, expected_radiance)
        
        self._metrics['train']['global']['loss'] += loss.item()
        self._metrics['train']['global']['r2'] += self._r2_metric(o_radiance, expected_radiance).item()
        
        loss.backward()
        
        self._optimizers['gnn'].step()
        self._optimizers['nerf'].step()
        
        
    def predict(self, data, scalers):
        
        # retrieve direct radiance
        camera_features = torch.cat([data.origin, data.direction], dim=1)
        y_direct_predicted = self._models['nerf'](camera_features).detach().cpu().numpy()
        
        # predict indirect
        y_predicted = self._models['gnn'](data)
        y_indirect_predicted = self._models['gnn'].radiance_from_predictions(y_predicted).detach().cpu().numpy()
        
        # Specific case: predicted encoded viewpoint information
        
        # Radiance (indirect and direct) must be the 3 thirds features to predict
        y_total_target = data.y_total.detach().cpu().numpy()
        input_radiance = data.direct_radiance.detach().cpu().numpy() + data.indirect_radiance.detach().cpu().numpy()
    
        # rescaled if necessary
        if scalers.get_scalers_from_field('y_total') is not None:
            y_total_target = scalers.inverse_transform_field('y_total', y_total_target)
            
        if scalers.get_scalers_from_field('y_direct') is not None:
            y_direct_predicted = scalers.inverse_transform_field('y_direct', y_direct_predicted)
            
        if scalers.get_scalers_from_field('y_indirect') is not None:
            y_indirect_predicted = scalers.inverse_transform_field('y_indirect', y_indirect_predicted)

        return input_radiance, (y_direct_predicted + y_indirect_predicted), y_total_target
    
    
    def score(self, mode, metric='r2'):
        
        # return global expected metric
        return self._metrics[mode]['global'][metric] / self._batchs[mode]
    
    
class ManagerFactory():
    
    @staticmethod
    def create(n_node_features, n_camera_features, config):
        
        # INSTANTIATE THE MODEL MANAGER
        model_list = []
        
        # instantiate GNN model
        gnn_params = {
            'graph_hidden_channels': config.GNN_HIDDEN_CHANNELS,
            'dense_hidden_layers': config.GNN_DENSE_HIDDEN,
            'n_dense_layers': config.GNN_N_DENSE_LAYERS,
            'latent_space': config.GNN_LATENT_SPACE,
            'n_features': n_node_features
        }
        
        # specific GNN kind
        if config.MODELS['gnn'] != "simple":
            gnn_params['n_camera_features'] = n_camera_features
        
        gnn_model_param = ModelParam(kind='gnn', name=config.MODELS['gnn'], loss=config.LOSS['gnn'], params=gnn_params)
        model_list.append(gnn_model_param)
        
        if config.TRAINING_MODE == 'separated':
            
            # Add NeRF model
            nerf_params = {
                'n_features': n_camera_features,
                'hidden_size': config.NERF_LAYER_SIZE,
                'n_hidden_layers': config.NERF_HIDDEN_LAYERS
            }
            
            nerf_model_param = ModelParam(kind='nerf', name=config.MODELS['nerf'], loss=config.LOSS['nerf'], params=nerf_params)
            model_list.append(nerf_model_param)
            
            return SeparatedModelManager(model_list)
        else:
            return SimpleModelManager(model_list)
