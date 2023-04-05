"""Abstract Graph Container"""
from abc import ABC, abstractmethod

from mignn.entities.graph.base import Graph

from typing import List
from itertools import chain

import numpy as np
import mitsuba as mi

class GraphContainer(ABC):
    """Abstract Graph class container (manage multiple graphs)"""
    
    def __init__(self):
        
        # track the number of built connection (and hence duplicate nodes)
        self._n_built_connections = 0
        self._n_built_nodes = 0
        self._graphs = {}
        
    @property
    def n_graphs(self) -> int:
        return sum([ len(v) for _, v in self._graphs.items() ])
    
    @property
    def n_nodes(self) -> int:
        return sum([ sum( [ len(g.nodes) for g in v]) \
            for _, v in self._graphs.items() ])
    
    @property
    def n_connections(self) -> int:
        return sum([ sum( [ len(g.connections) for g in v]) \
            for _, v in self._graphs.items() ])
    
    def keys(self) -> List[str]:
        return sorted(self._graphs.keys())
    
    def items(self) -> List[tuple[str, List[Graph]]]:
        return sorted(self._graphs.items())
    
    def graphs_at(self, pos: tuple[int, int]) -> List[Graph]:
        return self._graphs[pos]
    
    @property
    def graphs(self) -> List[Graph]:
        return list(chain(*[ v for _, v in self._graphs.items() ]))
    
    def add_graphs(self, pos: tuple[int, int], graphs: Graph) -> None:
        
        pos = tuple(pos)
        if isinstance(graphs, list):
            self._graphs[pos] += graphs
        
    def add_graph(self, pos: tuple[int, int], graph: Graph) -> None:
        
        pos = tuple(pos)
        
        if tuple(pos) not in self._graphs:
            self._graphs[pos] = []
            
        self._graphs[pos].append(graph)
    
    # TODO: do the same function but with convolution
    def build_connections(self, n_graphs: int, n_nodes_per_graphs: int, n_neighbors: int, \
        verbose: bool=False) -> None: 
        
        for idx, (key, _) in enumerate(self._graphs.items()):
            
            self._build_pos_connections(key, n_graphs, n_nodes_per_graphs, n_neighbors)
            
            if verbose:
                print(f'[Connections build] -- progress: {(idx + 1) / len(self.keys()) * 100.:.2f}%', \
                    end='\r' if idx + 1 < len(self.keys()) else '\n')
            
    
    def _init_graphs(self, init_graphs: dict) -> None:
        self._graphs = init_graphs
        
    @abstractmethod
    def _build_pos_connections(self, pos: tuple[int, int], n_graphs: int, \
        n_nodes_per_graphs: int, n_neighbors: int):
        """
        For each position from current film, new connections are tempted to be build:
        - n_graphs: number of graphs to update
        - n_nodes_per_graphs: expected number of nodes to get new connections (chosen randomly)
        - n_neighbors: number of neighbors graph to take in account 
        """
    
    @classmethod
    @abstractmethod
    def from_params(cls, container):
        pass
            
    @abstractmethod
    def _extract_light_grath(self, line: str) -> Graph:
        pass

    def _load_fromfile(self, filename: str, verbose: bool=True):
    
        with open(filename, 'r', encoding="utf-8") as f_light_path:

            lines = f_light_path.readlines()
            n_lines = len(lines)
            step = n_lines // 100
            for idx, line in enumerate(lines):

                pos, graph = self._extract_light_grath(line)
                self.add_graph(pos, graph)

                if verbose and (idx % step == 0 or idx >= n_lines - 1):
                    print(f'[Load of `{filename}`] -- progress: {(idx + 1) / n_lines * 100.:.2f}%', \
                          end='\r' if idx + 1 < n_lines else '\n')
    
    def __str__(self) -> str:
        return f'[n_keys: {len(self._graphs.keys())}, n_graphs: {self.n_graphs}, n_nodes: {self.n_nodes} ' \
            f'(duplicate: {self._n_built_nodes}), n_connections: {self.n_connections} ' \
            f'(built: {self._n_built_connections})]'
            
            

class LightGraphContainer(GraphContainer, ABC):
    
    def __init__(self, scene: mi.Scene, reference: np.ndarray=None, \
        variant: str='scalar_rgb'):
        
        super().__init__()
        self._scene = scene
        self._reference = reference
        self._mi_variant = variant   
        
    @property
    def scene(self) -> str:
        return self._scene
    
    @property
    def reference(self) -> np.ndarray:
        return self._reference
    
    @property
    def variant(self) -> str:
        return self._mi_variant
    
    @classmethod 
    def from_params(cls, container):
        
        # init same container with same expected keys but empty
        container_instance = cls(container.scene, container.reference, container.variant)
        empty_dict_keys = dict(zip(container.keys(), [ [] for _ in container.keys() ]))
        container_instance._init_graphs(empty_dict_keys)
        
        return container_instance
       
    @classmethod
    def fromfile(cls, filename: str, scene: mi.Scene, reference: np.ndarray=None, \
        variant: str='scalar_rgb', verbose: bool=True):

        graph_container = cls(scene, reference, variant)
        graph_container._load_fromfile(filename, verbose)
        
        return graph_container
