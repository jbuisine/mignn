"""Ray Graph module"""
from .base import Graph

class LightGraph(Graph):
    """Specific Ray Graph class"""
    def __init__(self, origin, luminance):
        super().__init__()
        self._origin = origin
        self._luminance = luminance
    
    @property
    def origin(self):
        return self._origin
    
    @property
    def luminance(self):
        return self._luminance
    
    def __str__(self):
        return f'Graph: [origin: {self._origin}, luminance: {self._luminance},' \
            f'nodes: {self._nodes}, connections: {self._connections}]'