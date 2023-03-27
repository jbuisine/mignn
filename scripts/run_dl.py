import os
import numpy as np

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

from mignn.container import SimpleLightGraphContainer
from mignn.manager import LightGraphManager

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from sklearn.preprocessing import MinMaxScaler
from torchmetrics import R2Score

from models.gcn_model import GNNL

DATASET_PATH = '../notebooks/data/datasets/cornell'

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


def load_sensor(r, phi, theta, target):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 25,
        # -1 0 0 0 0 1 0 1 0 0 -1 6.8 0 0 0 1
        'to_world': T.look_at(
            origin=origin,
            target=target,
            up=[0, 1, 0]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 10
        },
        'film': {
            'type': 'hdrfilm',
            'width': 64,
            'height': 64,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })


def prepare_data(scene_file, max_depth, data_spp, ref_spp, sensors, output_folder):
        
    os.makedirs(output_folder, exist_ok=True)
    
    scene = mi.load_file(scene_file)

    ref_integrator = mi.load_dict({'type': 'path', 'max_depth': max_depth})
    gnn_integrator = mi.load_dict({'type': 'pathgnn', 'max_depth': max_depth})
        
    # generate gnn file data and references
    ref_images = []
    output_gnn_files = []
    
    print(f'Generation of {len(sensors)} views for `{scene_file}`')
    for view_i, sensor in enumerate(sensors):
        
        print(f'Generating data for view n°{view_i+1}')
        ref_images.append(mi.render(scene, spp=ref_spp, integrator=ref_integrator, sensor=sensor))
        
        print(f' -- reference of view n°{view_i+1} generated...')
        params = mi.traverse(scene)
        gnn_log_filename = f'{output_folder}/gnn_file_{view_i}.path'
        params['logfile'] = gnn_log_filename
        params.update();
        
        if not os.path.exists(gnn_log_filename):
            mi.render(scene, spp=data_spp, integrator=gnn_integrator, sensor=sensor)
        print(f' -- GNN data of view n°{view_i+1} generated...')
        
        output_gnn_files.append(gnn_log_filename)
        
    return output_gnn_files, ref_images

def embedded_data(x_data, L=6):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    powers = torch.pow(2., torch.arange(L))
    
    x_data[np.isinf(x_data)] = 0
    emb_data = []
    
    for p in x_data:
        coord_data = torch.empty(0)
        for coord in p:
            x_cos = torch.cos(coord * powers)
            x_sin = torch.cos(coord * powers)
            coord_emb = torch.cat((coord.unsqueeze(0), x_cos, x_sin), 0)
            coord_data = torch.cat((coord_data, coord_emb), 0)
        emb_data.append(coord_data)
    
    return torch.stack(emb_data)

def main():
    sensor_count = 6

    radius = 5
    phis = [ 140 - (i*20) for i in range(sensor_count)]
    theta = 22

    sensors = [load_sensor(radius, phi, theta, [0, 1, 0]) for phi in phis]

    scene_file = '../notebooks/scenes/cornell-box/scene.xml'
    
    if not os.path.exists(DATASET_PATH):
        gnn_files, ref_images = prepare_data(scene_file, 
                                    max_depth = 5, 
                                    data_spp = 10, 
                                    ref_spp = 1000, 
                                    sensors = sensors[:1], 
                                    output_folder = 'data/model1')
        
        containers = []
        for gnn_i, gnn_file in enumerate(gnn_files):
            ref_image = ref_images[gnn_i]
            container = SimpleLightGraphContainer.fromfile(gnn_file, scene_file, ref_image, verbose=True)
            containers.append(container)
            
        # build connections individually
        for container in containers:
            container.build_connections(n_graphs=10, n_nodes_per_graphs=5, n_neighbors=5, verbose=True)
            
        merged_graph_container = LightGraphManager.fusion(containers)
        merged_graph_container = LightGraphManager.vstack(merged_graph_container)
        print(merged_graph_container)
        
        # prepare Dataset    
        data_list = []

        for key in merged_graph_container.keys():
            graphs = merged_graph_container.graphs_at(key)
            for graph in graphs:
                torch_data = graph.data.to_torch()
                
                # do embedding
                data = Data(x = embedded_data(torch_data.x), 
                        edge_index = torch_data.edge_index,
                        y = np.log(1 + torch_data.y),
                        edge_attr = embedded_data(torch_data.edge_attr),
                        pos = torch_data.pos)
                data_list.append(data)
                
        # save dataset
        PathLightDataset(DATASET_PATH, data_list)

    # transform applied only when loaded
    dataset = PathLightDataset(root=DATASET_PATH)
    print(dataset[0])

    split_index = int(len(dataset) * 0.8)
    train_dataset = dataset[:split_index]
    test_dataset = dataset[split_index:]
    
    # normalize data
    x_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_dataset.data.x)
    edge_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_dataset.data.edge_attr)
    y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_dataset.data.y)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    
    model = GNNL(hidden_channels=32, n_features=dataset.num_node_features)
    print(model)
    print(f'Number of params: {sum(p.numel() for p in model.parameters())}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.HuberLoss()
    r2 = R2Score()

    def train(epoch_id):
        model.train()

        error = 0
        r2_error = 0
        for b_i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
            
            # normalize data
            x_data = torch.tensor(x_scaler.transform(data.x), dtype=torch.float)
            x_edge_data = torch.tensor(edge_scaler.transform(data.edge_attr), dtype=torch.float)
            y_data = torch.tensor(y_scaler.transform(data.y), dtype=torch.float)
            # y_data = data.y
            
            out = model(x_data, x_edge_data, data.edge_index, batch=data.batch)  # Perform a single forward pass.
            loss = criterion(out, y_data)  # Compute the loss.
            r2_error += r2(out.flatten(), y_data.flatten())
            error += loss.item()
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            
            print(f'[Epoch n°{epoch_id}] -- progress: {(b_i + 1) / len(train_loader) * 100.:.2f}%' \
                f' (Loss: {error / (b_i + 1):.5f}, R²: {r2_error / (b_i + 1):.5f})', end='\r')

    def test(loader):
        model.eval()

        error = 0
        r2_error = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            
            # normalize data
            x_data = torch.tensor(x_scaler.transform(data.x), dtype=torch.float)
            x_edge_data = torch.tensor(edge_scaler.transform(data.edge_attr), dtype=torch.float)
            y_data = torch.tensor(y_scaler.transform(data.y), dtype=torch.float)
            # y_data = data.y
            
            out = model(x_data, x_edge_data, data.edge_index, batch=data.batch)
            loss = criterion(out, y_data)
            error += loss.item()  
            r2_error += r2(out.flatten(), y_data.flatten())
        return error / len(loader), r2_error / len(loader)  # Derive ratio of correct predictions.


    for epoch in range(1, 21):
        train(epoch)
        train_loss, train_r2 = test(train_loader)
        test_loss, test_r2 = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train (Loss: {train_loss:.5f}, R²: {train_r2:.5f}), '\
            f'Test (Loss: {test_loss:.5f}, R²: {test_r2:.5f})', end='\n')
            
    model_folder = '../notebooks/data/models/model1'
    os.makedirs('../notebooks/data/models/model1', exist_ok=True)
    torch.save(model.state_dict(), f'{model_folder}/model.pt')
    torch.save(optimizer.state_dict(), f'{model_folder}/optimizer.pt')
            
if __name__ == "__main__":
    main()
