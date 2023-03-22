"""Abstract Connection"""
from abc import ABC, abstractmethod
from typing import List

from mignn.entities.node.base import Node

class Connection(ABC):
    """Abstract Connection object class
    """
    
    def __init__(self, from_node: Node, to_node: Node, data: dict) -> None:
        
        self._from_node = from_node
        self._to_node = to_node
        self._data = data
    
    @property
    def from_node(self) -> Node:
        return self._from_node
    
    @property
    def to_node(self) -> Node:
        return self._to_node
    
    @property
    @abstractmethod
    def properties(self) -> List[float]:
        """Get all properties describing this node
        """
