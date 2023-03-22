"""Simple Ray Connection"""
from .base import Connection
from typing import List

from mignn.entities.node.base import Node

class RayConnection(Connection):
    """Ray Connection object class
    """
    
    def __init__(self, from_node: Node, to_node: Node, data: dict) -> None:
        super().__init__(from_node, to_node, data)
    
    @property
    def properties(self) -> List[float]:
        """Get all properties describing this node
        """
        return [ self._data['distance'] ]
