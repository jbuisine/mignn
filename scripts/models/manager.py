from abc import abstractmethod

from typing import List

import torch
from torch.autograd import Variable
from torchmetrics import R2Score

from .factory import ModelFactory
from .param import ModelParam

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
        pass
        
    @abstractmethod
    def step(self, data) -> dict:
        """Perform one train step from batch data and return 
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        pass
    
    def clear_metrics(self):
        self._init_metrics()
        
    def information(self, mode):

        if mode in ['train', 'test']:
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
        
        y_predicted = self._models['gnn'](data)
        y_target = data.y_direct.flatten() + data.y_indirect.flatten()
        
        # predict whole radiance
        loss = self._losses['gnn'](y_predicted.flatten(), y_target)
        self._metrics['test']['gnn']['loss'] += loss.item()
        
        self._metrics['test']['gnn']['r2'] += self._r2_metric(y_predicted.flatten(), y_target).item()
        
        self._batchs['test'] += 1
                    
    def step(self, data) -> dict:
        
        # TRAIN GNN
        self._optimizers['gnn'].zero_grad() 
        
        y_predicted = self._models['gnn'](data)
        y_target = data.y_direct.flatten() + data.y_indirect.flatten()
        
        # predict whole radiance
        loss = self._losses['gnn'](y_predicted.flatten(), y_target)
        self._metrics['train']['gnn']['loss'] += loss.item()
        
        loss.backward()
        self._optimizers['gnn'].step()
        
        self._metrics['train']['gnn']['r2'] += self._r2_metric(y_predicted.flatten(), y_target).item()
        
        self._batchs['train'] += 1
        
        
    def predict(self, data, scalers) -> List[torch.Tensor]:
        
        y_predicted = self._models['gnn'](data).detach().cpu().numpy()
        
        # Radiance (indirect and direct) must be the 3 thirds features to predict
        y_total_target = data.y_total.detach().cpu().numpy()
        input_radiance = data.direct_radiance.detach().cpu().numpy() + data.indirect_radiance.detach().cpu().numpy()
        
        # rescaled if necessary
        if scalers.get_scalers_from_field('y_total') is not None:
            y_predicted = scalers.inverse_transform_field('y_total', y_predicted)
            y_total_target = scalers.inverse_transform_field('y_total', y_total_target)
            input_radiance = scalers.inverse_transform_field('y_total', input_radiance)

        return input_radiance, y_predicted, y_total_target
        
class SeparatedModelManager(ModelManager):
    
    def __init__(self, params: list[ModelParam]) -> None:
        super().__init__(params)
            
        self._global_criterion = torch.nn.MSELoss().to(self._device)
            
    def _build_models(self):
                
        # instantiate all models with specific kind
        for model_params in self._params:
                
            kind = model_params.kind
        
            self._models[kind] = ModelFactory.from_params(model_params)
            self._optimizers[kind] = self._init_optimizer(self._models[kind], 'adam')
            self._losses[kind] = self._init_loss(model_params.loss)
        
    def test(self, data) -> dict:
        pass
                    
    def step(self, data) -> dict:
        
        # TRAIN NERF    
        self._optimizers['nerf'].zero_grad()
        
        camera_features = torch.cat([data.origin, data.direction], dim=1)
        
        o_direct_radiance = self._models['nerf'](camera_features)
        # o_direct_radiance = o_direct_radiance[:, :-1] * o_direct_radiance[:, -1]
        loss = self._losses['nerf'](o_direct_radiance.flatten(), data.y_direct.flatten())  # Compute the loss.
        nerf_error += loss.item()
        loss.backward()
        self._optimizers['nerf'].step()  
        
        # TRAIN GNN
        self._optimizers['gnn'].zero_grad() 
        
        # TODO: predict also for each graph the camera and direction position
        o_indirect_radiance = self._models['gnn'](data)
        
        loss = self._losses['gnn'](o_indirect_radiance.flatten(), data.y_indirect.flatten())
        gnn_error += loss.item()
        
        loss.backward()
        self._optimizers['gnn'].step()  
        
        expected_radiance = Variable(data.y_direct.flatten() + data.y_indirect.flatten(), requires_grad=True)
        o_radiance = Variable(o_direct_radiance.flatten() + o_indirect_radiance.flatten(), requires_grad=True)
        
        # Propagate NERF global error into the two networks
        self._optimizers['gnn'].zero_grad() 
        self._optimizers['nerf'].zero_grad() 
        
        loss = self._global_criterion(o_radiance, expected_radiance)
        global_error += loss.item()
        loss.backward()
        
        self._optimizers['gnn'].step()
        self._optimizers['nerf'].step()
        
    def predict(self, data, scalers):
        
        # retrieve direct radiance
        nerf_input = torch.cat([data.origin, data.direction], dim=1)
        y_direct_predicted = self._models['nerf'](nerf_input).detach().cpu().numpy()
        
        # predict indirect
        y_indirect_predicted = self._models['gnn'](data).detach().cpu().numpy()
        
        # Radiance (indirect and direct) must be the 3 thirds features to predict
        y_total_target = data.y_total.detach().cpu().numpy()
        input_radiance = data.direct_radiance.detach().cpu().numpy() + data.indirect_radiance.detach().cpu().numpy()
        
        # rescaled if necessary
        if scalers.get_scalers_from_field('y_total') is not None:
            y_total_target = scalers.inverse_transform_field('y_total', y_total_target)
            input_radiance = scalers.inverse_transform_field('y_total', input_radiance)
            
        if scalers.get_scalers_from_field('y_direct') is not None:
            y_direct_predicted = scalers.inverse_transform_field('y_direct', y_direct_predicted)
            
        if scalers.get_scalers_from_field('y_indirect') is not None:
            y_indirect_predicted = scalers.inverse_transform_field('y_indirect', y_indirect_predicted)

        return input_radiance, (y_direct_predicted + y_indirect_predicted), y_total_target