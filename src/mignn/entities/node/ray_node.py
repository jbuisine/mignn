"""Node implementation for RayGraph"""

from .base import Node

class RayNode(Node):
    """Ray Node class
    """
    
    def __init__(self, position: list[float, float, float], \
            normal: list[float, float, float]):
        super().__init__()
        self._position = position
        self._normal = normal
       
    @property
    def position(self):
        return self._position
    
    @property
    def normal(self):
        return self._normal
    
    @property
    def properties(self):
        return self._position + self._normal
    
    def __str__(self):
        return f'[position: {self.position}, normal: {self.normal}]'