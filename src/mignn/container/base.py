"""Abstract Graph Container"""
from abc import ABC, abstractmethod

from mignn.entities.graph.base import Graph

from typing import List
import numpy as np

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
        return self._graphs.keys()
    
    def items(self):
        return self._graphs.items()
    
    def get_graphs(self, pos) -> Graph:
        return self._graphs[pos]
    
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
        verbose: bool=False): 
        
        for idx, (key, _) in enumerate(self._graphs.items()):
            
            self._build_pos_connections(key, n_graphs, n_nodes_per_graphs, n_neighbors)
            
            if verbose:
                print(f'Connections build {(idx + 1) / len(self.keys()) * 100.:.2f}%', end='\r')
            
    @abstractmethod
    def _build_pos_connections(self, pos, n_graphs, n_nodes_per_graphs, n_neighbors):
        """
        For each position from current film, new connections are tempted to be build:
        - n_graphs: number of graphs to update
        - n_nodes_per_graphs: expected number of nodes to get new connections (chosen randomly)
        - n_neighbors: number of neighbors graph to take in account 
        """
    
    @classmethod
    @abstractmethod
    def params_copy(cls, container):
        pass
        
    @classmethod
    def _init_from_keys(cls, container):
        
        container_instance = cls()
        init_dict = dict(zip(container.keys(), [ [] for _ in container.keys() ]))
        
        # TODO: need to be improved
        container_instance._graphs = init_dict
        
        return container_instance
     
    @classmethod
    @abstractmethod
    def _extract_light_grath(cls, line):
        pass

    @classmethod
    def _load_fromfile(cls, filename: str, verbose: bool=True):
    
        graph_container = cls()

        with open(filename, 'r', encoding="utf-8") as f_light_path:

            lines = f_light_path.readlines()
            n_lines = len(lines)
            step = n_lines // 100
            for idx, line in enumerate(lines):

                pos, graph = cls._extract_light_grath(line)
                graph_container.add_graph(pos, graph)

                if verbose and (idx % step == 0 or idx >= n_lines - 1):
                    print(f'Load of `{filename}` in progress: {(idx + 1) / n_lines * 100.:.2f}%', \
                          end='\r' if idx + 1 < n_lines else '\n')
                
        return graph_container
    
    def __str__(self):
        return f'[n_keys: {len(self._graphs.keys())}, n_graphs: {self.n_graphs}, n_nodes: {self.n_nodes} ' \
            f'(duplicate: {self._n_built_nodes}), n_connections: {self.n_connections} ' \
            f'(built: {self._n_built_connections})]'
            
            

class LightGraphContainer(GraphContainer, ABC):
    
    def __init__(self, variant: str='scalar_rgb'):
        
        super().__init__()
        self._scene_file = None
        self._reference = None
        self._mi_variant = variant   
        
    @property
    def scene_file(self) -> str:
        return self._scene_file
    
    @property
    def reference(self) -> np.ndarray:
        return self._reference
    
    @property
    def variant(self) -> str:
        return self._mi_variant
    
    def __set_scene_file(self, scene_file: str):
        self._scene_file = scene_file
        
    def __set_reference(self, reference: np.ndarray):
        self._reference = reference
        
    def __set_variant(self, variant: str):
        self._mi_variant = variant
        
    @classmethod
    def params_copy(cls, container):
        
        container_instance = cls._init_from_keys(container)
        container_instance.__set_scene_file(container.scene_file)
        container_instance.__set_reference(container.reference)
        container_instance.__set_variant(container.variant)
        
        return container_instance
       
    @classmethod
    def fromfile(cls, filename: str, scene_file: str, reference: np.ndarray=None, \
        verbose: bool=True):
        
        graph_container = cls._load_fromfile(filename, verbose)
        graph_container.__set_scene_file(scene_file)
        graph_container.__set_reference(reference)
        
        return graph_container