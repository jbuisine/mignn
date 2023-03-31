"""Abstract Graph"""
from abc import ABC
from typing import List
from typing import NamedTuple

from mignn.entities.node.base import Node
from mignn.entities.connection.base import Connection

import torch

class GraphData(NamedTuple):
    x: List[float]
    edge_index: tuple[List[int], List[int]]
    edge_attr: List[float] 
    y: List[float]
    edge_tag: List[int]
    pos: List[tuple[float, float, float]] = None
    
    def to_torch(self):
        
        return GraphData(
            edge_index = torch.tensor(self.edge_index, dtype=torch.long),
            x = torch.tensor(self.x, dtype=torch.float),
            edge_attr = torch.tensor(self.edge_attr, dtype=torch.float),
            y = torch.tensor(self.y, dtype=torch.float),
            pos = torch.tensor(self.pos, dtype=torch.float),
            edge_tag = self.edge_tag
        )

class Graph(ABC):
    
    def __init__(self, targets: List[float]):
    
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
    def data(self) -> GraphData:
        # create prepare graph data
        edges_data = [ (self._nodes.index(con.from_node), \
                self._nodes.index(con.to_node), con.properties, con.tag) \
                for con in self._connections 
            ]
        edges_from, edges_to, edges_properties, edges_tags = list(zip(*edges_data))
        
        return GraphData(
            x = [ n.properties for n in self._nodes ],
            edge_index = [edges_from, edges_to],
            edge_attr = edges_properties,
            y = self._targets,
            edge_tag = edges_tags,
            pos = [ n.position for n in self._nodes ]
        )
        
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
    
    def __str__(self) -> str:
        return f'nodes: {self._nodes}, connections: {self._connections}]'