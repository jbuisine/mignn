import torch
from torch_geometric.data import InMemoryDataset

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
            return self.data.y_indirect.size()[-1]

        return None
    