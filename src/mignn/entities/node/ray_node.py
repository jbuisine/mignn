"""Node implementation for RayGraph"""

from .base import Node

from typing import List

class RayNode(Node):
    """Ray Node class
    """
    
    def __init__(self, position: list[float, float, float], \
            normal: list[float, float, float], primary: bool=False):
        super().__init__()
        self._position = position
        self._normal = normal
        self._primary = primary
       
    @property
    def position(self) -> List[float]:
        return self._position
    
    @property
    def normal(self) -> List[float]:
        return self._normal
    
    @property
    def properties(self) -> List[float]:
        return self._position + self._normal

    @property
    def primary(self) -> bool:
        return self._primary
        
    def __str__(self) -> str:
        return f'[position: {self.position}, normal: {self.normal}, \
            primary: {self._primary}]'