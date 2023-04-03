import torch

def signal_embedding(x_data, L=20):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    powers = torch.pow(2., torch.arange(L))
    
    # avoid infinite data
    x_data[torch.isinf(x_data)] = 0
    emb_data = []
    
    # for each data elements
    for data in x_data:
        
        out_data = torch.empty(0)
        
        for c_data in data:
            x_cos = torch.cos(c_data * powers)
            x_sin = torch.cos(c_data * powers)
            c_emb = torch.cat((c_data.unsqueeze(0), x_cos, x_sin), 0)
            out_data = torch.cat((out_data, c_emb), 0)
            
        emb_data.append(out_data)
    
    return torch.stack(emb_data)
