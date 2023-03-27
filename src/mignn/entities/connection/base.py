"""Abstract Connection"""
from abc import ABC, abstractmethod
from typing import List
from enum import Enum

from mignn.entities.node.base import Node

class ConnectionTag(Enum):
    ORIGINAL = 0
    BUILT = 1
    
class Connection(ABC):
    """Abstract Connection object class
    """
    
    def __init__(self, from_node: Node, to_node: Node, data: dict, \
        tag :ConnectionTag) -> None:
        
        self._from_node = from_node
        self._to_node = to_node
        self._data = data
        self._tag = tag
    
    @property
    def from_node(self) -> Node:
        return self._from_node
    
    @property
    def to_node(self) -> Node:
        return self._to_node
    
    @property
    def tag(self) -> ConnectionTag:
        return self._tag
    
    @property
    def data(self) -> dict:
        return self._data
    
    @property
    @abstractmethod
    def properties(self) -> List[float]:
        """Get all properties describing this node
        """
