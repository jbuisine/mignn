"""Ray Graph module"""
from .base import Graph

class LightGraph(Graph):
    """Specific Ray Graph class"""
    def __init__(self, origin, targets):
        super().__init__(targets)
        self._origin = origin
    
    @property
    def origin(self):
        return self._origin
    
    def __str__(self):
        return f'Graph: [origin: {self._origin}, targets: {self._targets},' \
            f'nodes: {self._nodes}, connections: {self._connections}]'