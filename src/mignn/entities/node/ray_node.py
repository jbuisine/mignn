"""Node implementation for RayGraph"""

from .base import Node

import numpy as np

from typing import List

class RayNode(Node):
    """Ray Node class
    """
    
    def __init__(self, position: list[float, float, float], \
            normal: list[float, float, float], \
            radiance: list[float, float, float], primary: bool=False):
        super().__init__()
        self._position = position
        self._normal = normal
        self._radiances = []
        self._radiances.append(radiance)
        self._primary = primary
       
    @property
    def position(self) -> List[float]:
        return self._position
    
    @property
    def normal(self) -> List[float]:
        return self._normal
    
    @property
    def radiance(self) -> List[float]:
        return list(np.mean(self._radiances, axis=0))
    
    @property
    def properties(self) -> List[float]:
        return self._position + self._normal + self.radiance

    @property
    def primary(self) -> bool:
        return self._primary
    
    def add_radiance(self, radiance: list[float, float, float]) -> None:
        self._radiances.append(radiance)
        
    def __str__(self) -> str:
        return f'[position: {self.position}, normal: {self.normal}, \
            radiance: {self.radiance}, primary: {self._primary}]'