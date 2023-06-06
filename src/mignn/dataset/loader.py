import torch
from torch_geometric.data import Data
 
class GraphLoader():
    
    @staticmethod
    def load(pixel, data):
        
        # TODO: AVOID empty graph (no node data)
            
        pixel = torch.tensor(pixel, dtype=torch.int32)
        origin = torch.tensor(data["origin"], dtype=torch.float)
        direction = torch.tensor(data["direction"], dtype=torch.float)
            
        # nodes data
        x_node = torch.tensor(data["x"], dtype=torch.float)
        x_node_pos = torch.tensor(data["pos"], dtype=torch.float)
        
        # edges data (need to check empty edge data)
        edge_index = [] if data["edge_index"] is None else data["edge_index"]
        edge_index = torch.tensor(edge_index, dtype=torch.long)    
        
        edge_attr = [] if data["edge_attr"] is None else data["edge_attr"]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # targets
        y_direct_targets = torch.tensor(data["y_direct"], dtype=torch.float)
        y_indirect_targets = torch.tensor(data["y_indirect"], dtype=torch.float)
        c_direct_radiance = torch.tensor(data["direct_radiance"], dtype=torch.float)
        c_indirect_radiance = torch.tensor(data["indirect_radiance"], dtype=torch.float)
        
        return Data(x=x_node, 
                    origin=origin,
                    direction=direction,
                    pos=x_node_pos,
                    edge_index=edge_index.t().contiguous(), 
                    edge_attr=edge_attr,
                    y_direct=y_direct_targets,
                    y_indirect=y_indirect_targets,
                    direct_radiance=c_direct_radiance,
                    indirect_radiance=c_indirect_radiance,
                    pixel=pixel)
            