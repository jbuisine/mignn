"""Simple Ray Connection"""
from .base import Connection, ConnectionTag
from typing import List

from mignn.entities.node.base import Node


class RayConnection(Connection):
    """Ray Connection object class
    """
    
    def __init__(self, from_node: Node, to_node: Node, data: dict, \
        tag :ConnectionTag) -> None:
        super().__init__(from_node, to_node, data, tag)
    
    @property
    def properties(self) -> List[float]:
        """Get all properties describing this node
        """
        return [self._data['distance']]
    
    def __str__(self):
        return f'[from: {self.from_node}, to: {self.to_node}, \
            properties: {self.properties}]'
