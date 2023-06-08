import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, EdgeCNN, SAGEConv
from torch_geometric.nn import GraphNorm, BatchNorm
from torch_geometric.nn import global_mean_pool

class GNNL(torch.nn.Module):
    def __init__(self, graph_hidden_channels, dense_hidden_layers, latent_size, n_dense_layers, n_features):
        
        super(GNNL, self).__init__()
        
        self.conv1 = GCNConv(n_features, graph_hidden_channels)
        self.norm1 = BatchNorm(graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.norm2 = BatchNorm(graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, latent_size)
        self.norm3 = BatchNorm(latent_size)
        
        self._linear_sequence = torch.nn.Sequential()
        
        self._linear_sequence.append(Linear(latent_size, dense_hidden_layers))
        self._linear_sequence.append(torch.nn.Dropout(0.5))
        self._linear_sequence.append(torch.nn.ReLU())
        
        for _ in range(n_dense_layers):
            self._linear_sequence.append(Linear(dense_hidden_layers, dense_hidden_layers))
            self._linear_sequence.append(torch.nn.Dropout(0.5))
            self._linear_sequence.append(torch.nn.ReLU())
            
        self._linear_sequence.append(Linear(dense_hidden_layers, 3))

    def forward(self, data):
        # 1. Obtain node embeddings 
        # x_edge = self.edge_conv1(x_node, edge_index, edge_attr=edge_attr)
        # x_edge = self.norm1(x_edge)
        # x_edge = x_edge.relu()
        # x_edge = self.edge_conv2(x_edge, edge_index, edge_attr=edge_attr)
        # x_edge = self.norm2(x_edge)
        # x_edge = x_edge.relu()
        # x_edge = self.edge_conv3(x_edge, edge_index, edge_attr=edge_attr)
        # x_edge = self.norm3(x_edge)
        
        x_node = self.conv1(data.x, data.edge_index)
        x_node = self.norm1(x_node)
        x_node = F.dropout(x_node, 0.5)
        x_node = x_node.relu()
        x_node = self.conv2(x_node, data.edge_index)
        x_node = self.norm2(x_node)
        x_node = F.dropout(x_node, 0.5)
        x_node = x_node.relu()
        x_node = self.conv3(x_node, data.edge_index)
        x_node = self.norm3(x_node)
        x_node = F.dropout(x_node, 0.5)

        # 2. Readout layer
        h_node = global_mean_pool(x_node, data.batch)  # [batch_size, hidden_channels]
        # h_edge = global_mean_pool(x_edge, batch)  # [batch_size, hidden_channels]

        # x_data = torch.cat([h_edge, h_node], axis=1)
        x_data = h_node
        
        # 3. Apply a final classifier
        return self._linear_sequence(x_data)


# GNN Model with concatenated viewpoint information 
class GNNL_VP(torch.nn.Module):
    
    def __init__(self, graph_hidden_layers, dense_hidden_layers, 
                n_dense_layers, latent_size, n_features, n_camera_features):
        
        super(GNNL_VP, self).__init__()
        
        
        self.conv1 = GCNConv(n_features, graph_hidden_layers)
        self.norm1 = BatchNorm(graph_hidden_layers)
        self.conv2 = GCNConv(graph_hidden_layers, graph_hidden_layers)
        self.norm2 = BatchNorm(graph_hidden_layers)
        self.conv3 = GCNConv(graph_hidden_layers, latent_size)
        self.norm3 = BatchNorm(latent_size)
        
        self._linear_sequence = torch.nn.Sequential()
        
        # concatenate latent space of Node Graph + camera coordinates informations
        self._linear_sequence.append(Linear(latent_size + n_camera_features, dense_hidden_layers))
        self._linear_sequence.append(torch.nn.Dropout(0.5))
        self._linear_sequence.append(torch.nn.ReLU())
        
        for _ in range(n_dense_layers):
            self._linear_sequence.append(Linear(dense_hidden_layers, dense_hidden_layers))
            self._linear_sequence.append(torch.nn.Dropout(0.5))
            self._linear_sequence.append(torch.nn.ReLU())
        
        self._linear_sequence.append(Linear(dense_hidden_layers, 3))
        
        # TODO: coordinate camera could also be predicted instead of concatenated

    def forward(self, data):
        
        
        camera_features = torch.cat([data.origin, data.direction], dim=1)
        
        # 1. Obtain node embeddings     
        x_node = self.conv1(data.x, data.edge_index)
        x_node = self.norm1(x_node)
        x_node = F.dropout(x_node, 0.5)
        x_node = x_node.relu()
        x_node = self.conv2(x_node, data.edge_index)
        x_node = self.norm2(x_node)
        x_node = F.dropout(x_node, 0.5)
        x_node = x_node.relu()
        x_node = self.conv3(x_node, data.edge_index)
        x_node = self.norm3(x_node)
        x_node = F.dropout(x_node, 0.5)

        # 2. Readout layer
        h_node = global_mean_pool(x_node, data.batch)  # [batch_size, latent_space]

        # x_data = torch.cat([h_edge, h_node], axis=1)
        x_data = torch.cat([h_node, camera_features], axis=1)
        
        # 3. Apply a final classifier
        return self._linear_sequence(x_data)


# GNN Model with concatenated RGB and viewpoint information to predict
class GNNL_VPP(torch.nn.Module):
    
    def __init__(self, graph_hidden_layers, dense_hidden_layers, 
                n_dense_layers, latent_size, n_features, n_camera_features):
        
        super(GNNL_VP, self).__init__()
        
        
        self.conv1 = GCNConv(n_features, graph_hidden_layers)
        self.norm1 = BatchNorm(graph_hidden_layers)
        self.conv2 = GCNConv(graph_hidden_layers, graph_hidden_layers)
        self.norm2 = BatchNorm(graph_hidden_layers)
        self.conv3 = GCNConv(graph_hidden_layers, latent_size)
        self.norm3 = BatchNorm(latent_size)
        
        self._linear_sequence = torch.nn.Sequential()
        
        # concatenate latent space of Node Graph + camera coordinates informations
        self._linear_sequence.append(Linear(latent_size, dense_hidden_layers))
        self._linear_sequence.append(torch.nn.Dropout(0.5))
        self._linear_sequence.append(torch.nn.ReLU())
        
        for _ in range(n_dense_layers):
            self._linear_sequence.append(Linear(dense_hidden_layers, dense_hidden_layers))
            self._linear_sequence.append(torch.nn.Dropout(0.5))
            self._linear_sequence.append(torch.nn.ReLU())
        
        self._linear_sequence.append(Linear(dense_hidden_layers, 3 + n_camera_features))
        
        # TODO: coordinate camera could also be predicted instead of concatenated

    def forward(self, data):
        
        # 1. Obtain node embeddings     
        x_node = self.conv1(data.x, data.edge_index)
        x_node = self.norm1(x_node)
        x_node = F.dropout(x_node, 0.5)
        x_node = x_node.relu()
        x_node = self.conv2(x_node, data.edge_index)
        x_node = self.norm2(x_node)
        x_node = F.dropout(x_node, 0.5)
        x_node = x_node.relu()
        x_node = self.conv3(x_node, data.edge_index)
        x_node = self.norm3(x_node)
        x_node = F.dropout(x_node, 0.5)

        # 2. Readout layer
        h_node = global_mean_pool(x_node, data.batch)  # [batch_size, latent_space]
        
        # 3. Apply a final classifier in order to predict RGB values and camera
        return self._linear_sequence(h_node)





