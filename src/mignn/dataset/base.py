import torch
from torch_geometric.data import InMemoryDataset, Data

from mignn.container import SimpleLightGraphContainer

class PathLightDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None):
        self.data_list = data_list
        super().__init__(root, transform, log=False)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])
        
    @staticmethod
    def from_container(container: SimpleLightGraphContainer, output_path: str, \
        verbose: bool=True):
        
        # prepare Dataset    
        data_list = []
        
        n_keys = len(container.keys())
        step = n_keys // 100
        
        for idx, (_, graphs) in enumerate(container.items()):
            
            # graphs = merged_graph_container.graphs_at(key)
            for graph in graphs:
                torch_data = graph.data.to_torch()
                
                # fix infinite values
                edge_attr = torch_data.edge_attr
                edge_attr[torch.isinf(torch_data.edge_attr)] = 0
                
                data = Data(x = torch_data.x, 
                        edge_index = torch_data.edge_index,
                        y = torch_data.y,
                        edge_attr = edge_attr,
                        pos = torch_data.pos)
                
                data_list.append(data)
            
            if verbose and (idx % step == 0 or idx >= n_keys - 1):
                print(f'[Prepare torch data] progress: {(idx + 1) / n_keys * 100.:.0f}%', end='\r')
        
        # save dataset
        return PathLightDataset(output_path, data_list)

