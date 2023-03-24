"""Ray Graph module"""
from .base import Graph

from typing import List

class LightGraph(Graph):
    """Specific Ray Graph class"""
    def __init__(self, origin: tuple[float, float, float], targets: List[float]):
        super().__init__(targets)
        self._origin = origin
    
    @property
    def origin(self) -> List[float]:
        return self._origin
    
    def __str__(self) -> str:
        return f'Graph: [origin: {self._origin}, targets: {self._targets},' \
            f'nodes: {self._nodes}, connections: {self._connections}]'