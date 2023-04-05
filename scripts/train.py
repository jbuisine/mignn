import os
import numpy as np
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import uuid
import random

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

from mignn.container import SimpleLightGraphContainer
from mignn.manager import LightGraphManager
from mignn.dataset import PathLightDataset
from mignn.processing.encoder import signal_encoder

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump as skdump
from torchmetrics import R2Score

from models.gcn_model import GNNL

w_size, h_size = 16, 16
encoder_size = 6

def load_sensor_from(fov, origin, target, up):
    
    return mi.load_dict({
        'type': 'perspective',
        'fov': fov,
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
        
        # print(f'Generating data for view n°{view_i+1}')
        image = mi.render(scene, spp=ref_spp, integrator=ref_integrator, sensor=sensor)
        
        # save image as exr and reload it using cv2
        image_path = f'/tmp/{str(uuid.uuid4())}.exr'
        cv2.imwrite(image_path, np.asarray(image))
        exr_image = np.asarray(cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        ref_images.append(exr_image)
        
        # print(f' -- reference of view n°{view_i+1} generated...')
        params = mi.traverse(scene)
        gnn_log_filename = f'{output_folder}/gnn_file_{view_i}.path'
        params['logfile'] = gnn_log_filename
        params.update();
        
        if not os.path.exists(gnn_log_filename):
            mi.render(scene, spp=data_spp, integrator=gnn_integrator, sensor=sensor)
        print(f'[Generation] GNN data progress: {(view_i+1) / len(sensors) * 100:.2f}%', end='\r')
        
        output_gnn_files.append(gnn_log_filename)
        
    return output_gnn_files, ref_images

def main():
    
    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--output', type=str, help="output folder", required=True)
    parser.add_argument('--name', type=str, help="output model name", required=True)
    parser.add_argument('--epochs', type=int, help="expected number of epochs", required=False, default=10)
    parser.add_argument('--encoder', type=int, help="encoding data or not", required=False, default=False)
    parser.add_argument('--sensors', type=str, help="file with all viewpoints on scene", required=True)
    parser.add_argument('--split', type=float, help="split percent \in [0, 1]", required=False, default=0.8)
    
    args = parser.parse_args()
    
    scene_file        = args.scene
    output_folder     = args.output
    model_name        = args.name
    n_epochs          = args.epochs
    encoder_enabled   = args.encoder
    split_percent     = args.split
    sensors_folder    = args.sensors
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use of: https://github.com/prise-3d/vpbrt
    # read from camera LookAt folder
    sensors = []
    for file in os.listdir(sensors_folder):
        file_path = os.path.join(sensors_folder, file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            
            look_at_data = f.readline().replace('\n', '').split('  ')
            fov = float(f.readline().replace('\n', '').split(' ')[-1])
            origin = list(map(float, look_at_data[1].split(' ')))
            target = list(map(float, look_at_data[2].split(' ')))
            up = list(map(float, look_at_data[3].split(' ')))
            
            # print(f'LookAt: [origin: {origin}, target: {target}, up: {up}], Fov: {fov}')
            sensor = load_sensor_from(fov, origin, target, up)
            sensors.append(sensor)

    os.makedirs(output_folder, exist_ok=True)
    dataset_path = f'{output_folder}/train/datasets/{model_name}'
    
    if not os.path.exists(dataset_path):
        gnn_files, ref_images = prepare_data(scene_file, 
                                    max_depth = 5, 
                                    data_spp = 10, 
                                    ref_spp = 10000, 
                                    sensors = sensors, 
                                    output_folder = f'{output_folder}/train/generated/{model_name}')
        
        build_containers = []
        for gnn_i, gnn_file in enumerate(gnn_files):
            print(f'[Loading files] GNN data files: {(gnn_i + 1) / len(gnn_files) * 100:.2f}%', end="\r")
            ref_image = ref_images[gnn_i]
            container = SimpleLightGraphContainer.fromfile(gnn_file, scene_file, ref_image, verbose=True)
            
            container.build_connections(n_graphs=10, n_nodes_per_graphs=5, n_neighbors=5, verbose=True)
            build_container = LightGraphManager.vstack(container)
            build_containers.append(build_container)
            del container
            
        merged_graph_container = LightGraphManager.fusion(build_containers)
        print('[merged]', merged_graph_container)
        
        # prepare Dataset    
        data_list = []

        for kid, (_, graphs) in enumerate(merged_graph_container.items()):
            
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
                
            print(f'[Prepare torch data] progress: {(kid + 1) / len(merged_graph_container.keys()) * 100.:.2f}%', end='\r')
        
        # save dataset
        print(f'[Intermediate save] save computed dataset into: {dataset_path}')
        PathLightDataset(dataset_path, data_list)
    
    dataset = PathLightDataset(root=dataset_path)
    print(f'Dataset with {len(dataset)} graphs (percent split: {split_percent})')
    
    split_index = int(len(dataset) * split_percent)
    train_dataset = dataset[:split_index]
    test_dataset = dataset[split_index:]
    
    model_folder = f'{output_folder}/models/{model_name}'
    stats_folder = f'{output_folder}/stats/{model_name}'
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)
    
    # normalize data
    x_scaler = MinMaxScaler().fit(train_dataset.data.x)
    edge_scaler = MinMaxScaler().fit(train_dataset.data.edge_attr)
    # y_scaler = MinMaxScaler().fit(train_dataset.data.y.reshape((-1, 3)))
    
    skdump(x_scaler, f'{model_folder}/x_node_scaler.bin', compress=True)
    skdump(edge_scaler, f'{model_folder}/x_edge_scaler.bin', compress=True)
    # skdump(y_scaler, f'{model_folder}/y_scaler.bin', compress=True)
    
    if encoder_enabled:
        print('[Encoded required] scaled data will be encoded')
    
    scaled_dataset_path = f'{output_folder}/train/datasets/{model_name}_scaled'
        
    if not os.path.exists(scaled_dataset_path):
        
        scaled_data_list = []
        
        n_graphs = len(dataset)
        for d_i in range(n_graphs):
            
            data = dataset[d_i]
            
            # perform scale and then encoding
            x_data = torch.tensor(x_scaler.transform(data.x), dtype=torch.float)
            x_edge_data = torch.tensor(edge_scaler.transform(data.edge_attr), dtype=torch.float)
            
            if encoder_enabled:
                x_data = signal_encoder(x_data, L=encoder_size)
                x_edge_data = signal_encoder(x_edge_data, L=encoder_size)
                
            scaled_data = Data(x = x_data, 
                    edge_index = data.edge_index,
                    y = data.y,
                    edge_attr = x_edge_data,
                    pos = data.pos)
            
            scaled_data_list.append(scaled_data)
            
            print(f'[Prepare encoded torch data] progress: {(d_i + 1) / n_graphs * 100.:.2f}%', end='\r')
            
        # save dataset
        print(f'Save scaled dataset into: {scaled_dataset_path}')
        PathLightDataset(scaled_dataset_path, scaled_data_list)

    print(f'Load scaled dataset from: {scaled_dataset_path}')
    dataset = PathLightDataset(root=scaled_dataset_path)
    print(f'Example of scaled element from dataset: {dataset[0]}')
    
    split_index = int(len(dataset) * split_percent)
    train_dataset = dataset[:split_index]
    test_dataset = dataset[split_index:]
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    
    print('Prepare model: ')
    model = GNNL(hidden_channels=256, n_features=dataset.num_node_features)
    # model.to(device)
    print(model)
    print(f'Number of params: {sum(p.numel() for p in model.parameters())}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    r2_score = R2Score()

    def train(epoch_id):
        model.train()

        error = 0
        r2_error = 0
        for b_i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
            
            out = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)  # Perform a single forward pass.
            loss = criterion(out.flatten(), data.y)  # Compute the loss.
            error += loss.item()
            loss.backward()  # Derive gradients.
            r2_error += r2_score(out.flatten(), data.y.flatten())
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            
            print(f'[Epoch n°{epoch_id:03d}] -- progress: {(b_i + 1) / len(train_loader) * 100.:.2f}%' \
                f' (Loss: {error / (b_i + 1):.5f}, R²: {r2_error / (b_i + 1):.5f})', end='\r')

    def test(loader):
        model.eval()

        error = 0
        r2_error = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            
            out = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)
            loss = criterion(out.flatten(), data.y)
            error += loss.item()  
            r2_error += r2_score(out.flatten(), data.y.flatten())
        return error / len(loader), r2_error / len(loader)  # Derive ratio of correct predictions.

    stat_file = open(f'{stats_folder}/{model_name}.csv', 'w', encoding='utf-8')
    stat_file.write('train_loss;train_r2;test_loss;test_r2\n')

    for epoch in range(1, n_epochs + 1):
        train(epoch)
        train_loss, train_r2 = test(train_loader)
        test_loss, test_r2 = test(test_loader)
        
        # save model stat data
        stat_file.write(f'{train_loss};{train_r2};{test_loss};{test_r2}\n')
        
        print(f'[Epoch n°{epoch:03d}]: Train (Loss: {train_loss:.5f}, R²: {train_r2:.5f}), '\
            f'Test (Loss: {test_loss:.5f}, R²: {test_r2:.5f})', end='\n')
    
    stat_file.close()
            
    torch.save(model.state_dict(), f'{model_folder}/model.pt')
    torch.save(optimizer.state_dict(), f'{model_folder}/optimizer.pt')
    
    print(f'Model has been saved into: `{model_folder}`')
            
if __name__ == "__main__":
    main()
