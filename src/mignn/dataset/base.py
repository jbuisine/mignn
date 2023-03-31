import torch
from torch_geometric.data import InMemoryDataset

class PathLightDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
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

