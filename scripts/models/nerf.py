import torch
from torch.nn import Linear, Sequential
import torch.nn.functional as F

class BasicNeRf(torch.nn.Module):
    
    def __init__(self, n_features, hidden_size, n_hidden_layer):
        
        super(BasicNeRf, self).__init__()
        
        self._input = Linear(n_features, hidden_size)
        
        self.lin = []
        
        for _ in range()
        self._linear1 = Linear(hidden_size, hidden_size)
        self._linear2 = Linear(hidden_size, hidden_size)
        self._linear3 = Linear(hidden_size, hidden_size)
        self._linear4 = Linear(hidden_size, hidden_size)
        self._linear5 = Linear(hidden_size, hidden_size)
        self._linear6 = Linear(hidden_size, hidden_size)
        self._linear7 = Linear(hidden_size, hidden_size)
        self._output = Linear(hidden_size, 3)

    def forward(self, x):

        # TODO: use a dynamic way (here we avoid CUDA issue)
        x = self._input(x)
        x = x.relu()
        x = F.dropout(x, 0.5)
        
        x = self._linear1(x)
        x = x.relu()
        x = F.dropout(x, 0.5)
        
        x = self._linear2(x)
        x = x.relu()
        x = F.dropout(x, 0.5)
        
        x = self._linear3(x)
        x = x.relu()
        x = F.dropout(x, 0.5)
        
        x = self._linear4(x)
        x = x.relu()
        x = F.dropout(x, 0.5)
        
        x = self._linear5(x)
        x = x.relu()
        x = F.dropout(x, 0.5)
        
        x = self._linear6(x)
        x = x.relu()
        x = F.dropout(x, 0.5)
        
        x = self._linear7(x)
        x = x.relu()
        x = F.dropout(x, 0.5)
            
        return self._output(x)
