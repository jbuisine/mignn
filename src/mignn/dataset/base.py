import torch
from torch_geometric.data import InMemoryDataset, Data

from mignn.container import SimpleLightGraphContainer

class PathLightDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None, load=True):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform, log=False)
        
        if load:
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
        
        processed_data_list = []
        if self.pre_transform is not None:
            
            # TODO: check if possible to do chunks
            for i, _ in enumerate(self.data_list):
                processed_data_list.append(self.pre_transform(self.data_list[i]))
                    
            torch.save(self.collate(processed_data_list), self.processed_paths[0])
            
        else:
            torch.save(self.collate(self.data_list), self.processed_paths[0])
        
    @property         
    def num_target_features(self):
        if self.data is not None:
            return self.data.y.size()[-1]

        return None
    
    @staticmethod
    def from_container(container: SimpleLightGraphContainer, output_path: str, \
        load: bool=True, verbose: bool=True):
        
        # prepare Dataset    
        data_list = []
        
        n_keys = len(container.keys())
        step = (n_keys // 100) + 1
        
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
        return PathLightDataset(output_path, data_list, load=load)

