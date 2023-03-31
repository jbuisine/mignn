import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, EdgeCNN, SAGEConv
from torch_geometric.nn import global_mean_pool

class GNNL(torch.nn.Module):
    def __init__(self, hidden_channels, n_features):
        
        super(GNNL, self).__init__()
        # self.conv1 = EdgeCNN(n_features, hidden_channels, num_layers=10, dropout=0.4)
        # self.conv2 = EdgeCNN(hidden_channels, hidden_channels, num_layers=10, dropout=0.4)
        # self.conv3 = EdgeCNN(hidden_channels, hidden_channels, num_layers=10, dropout=0.4)
        
        self.conv1 = GCNConv(n_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.lin1 = Linear(hidden_channels, int(hidden_channels / 2))
        self.lin2 = Linear(int(hidden_channels / 2), 3)

    def forward(self, x, edge_attr, edge_index, batch):
        # 1. Obtain node embeddings 
        # x_edge = self.conv1(x, edge_index, edge_attr=edge_attr)
        # x_edge = x_edge.relu()
        # x_edge = self.conv2(x_edge, edge_index, edge_attr=edge_attr)
        # x_edge = x_edge.relu()
        # x_edge = self.conv3(x_edge, edge_index, edge_attr=edge_attr)
        
        x_node = self.conv1(x, edge_index)
        x_node = x_node.relu()
        x_node = self.conv2(x_node, edge_index)
        x_node = x_node.relu()
        x_node = self.conv3(x_node, edge_index)

        # 2. Readout layer
        h_node = global_mean_pool(x_node, batch)  # [batch_size, hidden_channels]
        # h_edge = global_mean_pool(x_edge, batch)  # [batch_size, hidden_channels]

        # x_data = torch.cat([h_edge, h_node], axis=1)
        x_data = h_node
        
        # 3. Apply a final classifier
        x = F.dropout(x_data, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x
