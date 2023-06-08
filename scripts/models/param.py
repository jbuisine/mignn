
class ModelParam():
    
    def __init__(self, kind: str, name: str, loss: str, params: dict) -> None:
        self._kind = kind
        self._name = name
        self._loss = loss
        self._params = params
        
    @property
    def kind(self):
        return self._kind
    
    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return self._params
    
    @property
    def loss(self):
        return self._loss

