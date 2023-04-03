import os
import numpy as np
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import uuid
import time

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

from mignn.container import SimpleLightGraphContainer
from mignn.manager import LightGraphManager
from mignn.dataset import PathLightDataset
from mignn.processing.embedding import signal_embedding

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump as skdump
from torchmetrics import R2Score

from models.gcn_model import GNNL

w_size, h_size = 128, 128

def load_sensor(r, phi, theta, target):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])
    up = [0, 1, 0]
    
    # matrix = mi.Transform4f.look_at(origin=origin, target=target, up=up)
    # print(matrix)
    
    return mi.load_dict({
        'type': 'perspective',
        'fov': 25,
        'to_world': T.look_at(
            origin=origin,
            target=target,
            up=up
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 10
        },
        'film': {
            'type': 'hdrfilm',
            'width': w_size,
            'height': h_size,
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
        image = mi.render(scene, spp=ref_spp, integrator=ref_integrator, sensor=sensor)
        
        # save image as exr and reload it using cv2
        image_path = f'/tmp/{str(uuid.uuid4())}.exr'
        cv2.imwrite(image_path, np.asarray(image))
        exr_image = np.asarray(cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        ref_images.append(exr_image)
        
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

def main():
    
    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--output', type=str, help="output model name", required=True)
    parser.add_argument('--epochs', type=int, help="expected number of epochs", required=False, default=10)
    parser.add_argument('--embedding', type=int, help="embedding data or not", required=False, default=False)
    parser.add_argument('--sensors', type=int, help="number of viewpoints on scene", required=False, default=6)
    parser.add_argument('--split', type=float, help="split percent \in [0, 1]", required=False, default=0.8)
    
    args = parser.parse_args()
    
    scene_file        = args.scene
    output_name       = args.output
    n_epochs          = args.epochs
    embedding_enabled = args.embedding
    sensor_count      = args.sensors
    split_percent     = args.split
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: add this part into library
    radius = 5
    phis = [ 140 - (i*20) for i in range(sensor_count)]
    theta = 22

    sensors = [load_sensor(radius, phi, theta, [0, 1, 0]) for phi in phis]

    dataset_path = f'data/train/datasets/{output_name}'
    
    if not os.path.exists(dataset_path):
        gnn_files, ref_images = prepare_data(scene_file, 
                                    max_depth = 5, 
                                    data_spp = 10, 
                                    ref_spp = 1000, 
                                    sensors = sensors, 
                                    output_folder = f'data/train/generated/{output_name}')
        
        containers = []
        for gnn_i, gnn_file in enumerate(gnn_files):
            ref_image = ref_images[gnn_i]
            container = SimpleLightGraphContainer.fromfile(gnn_file, scene_file, ref_image, verbose=True)
            containers.append(container)
            
        # build connections individually
        build_containers = []
        for container in containers:
            container.build_connections(n_graphs=10, n_nodes_per_graphs=5, n_neighbors=5, verbose=True)
            build_container = LightGraphManager.vstack(container)
            build_containers.append(build_container)
            
        merged_graph_container = LightGraphManager.fusion(build_containers)
        print('[merged]', merged_graph_container)
        
        # prepare Dataset    
        data_list = []

        for kid, (_, graphs) in enumerate(merged_graph_container.items()):
            
            # graphs = merged_graph_container.graphs_at(key)
            for graph in graphs:
                torch_data = graph.data.to_torch()
                
                # do embedding_enabled
                if embedding_enabled:
                    data = Data(x = signal_embedding(torch_data.x), 
                            edge_index = torch_data.edge_index,
                            y = torch_data.y,
                            edge_attr = signal_embedding(torch_data.edge_attr),
                            pos = torch_data.pos)
                else:
                    
                    edge_attr = torch_data.edge_attr
                    edge_attr[torch.isinf(torch_data.edge_attr)] = 0
    
                    data = Data(x = torch_data.x, 
                            edge_index = torch_data.edge_index,
                            y = torch_data.y,
                            edge_attr = edge_attr,
                            pos = torch_data.pos)
                    
                data_list.append(data)
                
            print(f'[Prepare torch data] progress: {(kid + 1) / len(merged_graph_container.keys()) * 100.:.2f}%', end='\r')
        
        # save dataset
        print(f'Save computed dataset into: {dataset_path}')
        PathLightDataset(dataset_path, data_list)

    # transform applied only when loaded
    print(f'Load dataset from: {dataset_path}')
    dataset = PathLightDataset(root=dataset_path)
    print(f'Example element from dataset: {dataset[0]}')

    split_index = int(len(dataset) * split_percent)
    train_dataset = dataset[:split_index]
    test_dataset = dataset[split_index:]
    
    model_folder = f'data/models/{output_name}'
    os.makedirs(model_folder, exist_ok=True)
    
    # normalize data
    print(f'Save scalers into: {model_folder}/scalers.pkl')
    x_scaler = MinMaxScaler().fit(train_dataset.data.x)
    edge_scaler = MinMaxScaler().fit(train_dataset.data.edge_attr)
    # y_scaler = MinMaxScaler().fit(train_dataset.data.y.reshape((-1, 3)))
    
    skdump(x_scaler, f'{model_folder}/x_node_scaler.bin', compress=True)
    skdump(edge_scaler, f'{model_folder}/x_edge_scaler.bin', compress=True)
    # skdump(y_scaler, f'{model_folder}/y_scaler.bin', compress=True)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    
    print('Prepare model: ')
    model = GNNL(hidden_channels=256, n_features=dataset.num_node_features)
    # model.to(device)
    print(model)
    print(f'Number of params: {sum(p.numel() for p in model.parameters())}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.L1Loss()
    r2 = R2Score()

    def train(epoch_id):
        model.train()

        error = 0
        r2_error = 0
        for b_i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
            
            # normalize data
            x_data = torch.tensor(x_scaler.transform(data.x), dtype=torch.float)
            x_edge_data = torch.tensor(edge_scaler.transform(data.edge_attr), dtype=torch.float)
            # y_data = torch.tensor(y_scaler.transform(data.y.reshape(-1, 3)), dtype=torch.float)
            y_data = data.y
            
            out = model(x_data, x_edge_data, data.edge_index, batch=data.batch)  # Perform a single forward pass.
            loss = criterion(out, y_data)  # Compute the loss.
            error += loss.item()
            loss.backward()  # Derive gradients.
            r2_error += r2(out.flatten(), y_data.flatten())
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            
            print(f'[Epoch n°{epoch_id:03d}] -- progress: {(b_i + 1) / len(train_loader) * 100.:.2f}%' \
                f' (Loss: {error / (b_i + 1):.5f}, R²: {r2_error / (b_i + 1):.5f})', end='\r')

    def test(loader):
        model.eval()

        error = 0
        r2_error = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            
            # normalize data
            x_data = torch.tensor(x_scaler.transform(data.x), dtype=torch.float)
            x_edge_data = torch.tensor(edge_scaler.transform(data.edge_attr), dtype=torch.float)
            # y_data = torch.tensor(y_scaler.transform(data.y.reshape(-1, 3)), dtype=torch.float)
            y_data = data.y
            
            out = model(x_data, x_edge_data, data.edge_index, batch=data.batch)
            loss = criterion(out, y_data)
            error += loss.item()  
            r2_error += r2(out.flatten(), y_data.flatten())
        return error / len(loader), r2_error / len(loader)  # Derive ratio of correct predictions.


    for epoch in range(1, n_epochs + 1):
        train(epoch)
        train_loss, train_r2 = test(train_loader)
        test_loss, test_r2 = test(test_loader)
        print(f'[Epoch: {epoch:03d}]: Train (Loss: {train_loss:.5f}, R²: {train_r2:.5f}), '\
            f'Test (Loss: {test_loss:.5f}, R²: {test_r2:.5f})', end='\n')
            
    torch.save(model.state_dict(), f'{model_folder}/model.pt')
    torch.save(optimizer.state_dict(), f'{model_folder}/optimizer.pt')
            
if __name__ == "__main__":
    main()
