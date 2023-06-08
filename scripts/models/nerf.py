import torch
from torch.nn import Linear, Sequential

class SimpleNeRf(torch.nn.Module):
    
    def __init__(self, n_features, hidden_size, n_hidden_layers):
        
        super(SimpleNeRf, self).__init__()
        
        self._sequence = Sequential()
        # input
        self._sequence.append(Linear(n_features, hidden_size))
        self._sequence.append(torch.nn.Dropout(0.5))
        self._sequence.append(torch.nn.ReLU())
        
        for _ in range(n_hidden_layers):
            self._sequence.append(Linear(hidden_size, hidden_size))
            self._sequence.append(torch.nn.Dropout(0.5))
            self._sequence.append(torch.nn.ReLU())
            
        self._sequence.append(Linear(hidden_size, 3))

    def forward(self, x):
    
        return self._sequence(x)
