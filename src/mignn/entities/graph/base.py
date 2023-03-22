"""Abstract Graph"""
from abc import ABC
from typing import List

from mignn.entities.node.base import Node
from mignn.entities.connection.base import Connection

class Graph(ABC):
    
    def __init__(self, targets):
    
        self._nodes = []
        self._connections = []
        self._targets = targets
    
    @property
    def nodes(self) -> List[Node]:
        return self._nodes
    
    @property
    def connections(self) -> List[Node]:
        return self._connections
    
    @property
    def targets(self) -> List[float]:
        return self._targets
    
    @property
    def data(self) -> tuple[List[List[float]], List[Connection]]:
        # create prepare graph data
        return {
            'nodes': [ n.properties for n in self._nodes ], 
            'edges': [ (self._nodes.index(con.from_node), \
                self._nodes.index(con.to_node), con.properties) \
                for con in self._connections 
            ],
            'targets': self._targets
        }
        
    
    def get_node_by_index(self, index) -> Node:
        
        if index < len(self._nodes):
            return self._nodes[index]
        return None
    
    def get_node_index(self, node) -> int:
        
        return self._nodes.index(node)
    
    def get_connections(self, node) -> List[Connection]:
        
        if node not in self._nodes:
            
            return list(filter(lambda c: c.from_node == node or c.to_node == node, \
                self._connections))
            
        return None
    
    def get_connections_from(self, node) -> List[Connection]:
        
        if node not in self._nodes:
            
            return list(filter(lambda c: c.from_node == node, \
                self._connections))
            
        return None
    
    def get_connections_to(self, node) -> List[Connection]:
        
        if node not in self._nodes:
            
            return list(filter(lambda c: c.to_node == node, \
                self._connections))
            
        return None
        
    def add_node(self, node) -> bool:
        
        if node not in self._nodes:
            self._nodes.append(node)
            return True
        return False
        
    def add_connection(self, connection: Connection) -> bool:
        
        connection_exist = len(list(filter(lambda c: c.from_node == connection.from_node \
                and c.to_node == connection.to_node, \
                self._connections)))
        
        if not connection_exist:
            self._connections.append(connection)    
            return True
        
        return False
    
    def __str__(self):
        return f'nodes: {self._nodes}, connections: {self._connections}]'