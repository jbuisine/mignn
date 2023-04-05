import os
import argparse
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

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

from joblib import load as skload

from models.gcn_model import GNNL

w_size, h_size = 32, 32
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

def prepare_data(scene_file, max_depth, data_spp, ref_spp, sensor, output_folder):
        
    os.makedirs(output_folder, exist_ok=True)
    
    scene = mi.load_file(scene_file)

    ref_integrator = mi.load_dict({'type': 'path', 'max_depth': max_depth})
    gnn_integrator = mi.load_dict({'type': 'pathgnn', 'max_depth': max_depth})
        
    # generate gnn file data and references
    print(f'Generating data for `{scene_file}`:')
    ref_image = mi.render(scene, spp=ref_spp, integrator=ref_integrator, sensor=sensor)
    
    print(' -- reference generated...')
    params = mi.traverse(scene)
    gnn_log_filename = f'{output_folder}/gnn_file_predict.path'
    params['logfile'] = gnn_log_filename
    params.update();
    
    if not os.path.exists(gnn_log_filename):
        mi.render(scene, spp=data_spp, integrator=gnn_integrator, sensor=sensor)
    print(f' -- GNN data generated...')
    
    output_gnn_file = gnn_log_filename
    
    return output_gnn_file, ref_image

def main():
    
    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--folder', type=str, help="main data folder (where to find model)", required=True)
    parser.add_argument('--name', type=str, help="model name", required=True)
    parser.add_argument('--outfile', type=str, help="output image name", required=True)
    parser.add_argument('--encoder', type=int, help="encoding data or not", required=False, default=False)
    parser.add_argument('--sensor', type=str, help="specific sensor file", required=True)
    
    args = parser.parse_args()
    
    scene_file        = args.scene
    main_folder       = args.folder
    model_name        = args.name
    outfile_name      = args.outfile
    encoder_enabled   = args.encoder
    sensor_file       = args.sensor
    
    output_name = outfile_name.split('.')[0]
    
    # use of: https://github.com/prise-3d/vpbrt
    # read from camera LookAt folder
    with open(sensor_file, 'r', encoding='utf-8') as f:
        
        look_at_data = f.readline().replace('\n', '').split('  ')
        fov = float(f.readline().replace('\n', '').split(' ')[-1])
        origin = list(map(float, look_at_data[1].split(' ')))
        target = list(map(float, look_at_data[2].split(' ')))
        up = list(map(float, look_at_data[3].split(' ')))
        
        print(f'LookAt: [origin: {origin}, target: {target}, up: {up}], Fov: {fov}')
        sensor = load_sensor_from(fov, origin, target, up)
    
    dataset_path = f'{main_folder}/predictions/datasets/{output_name}'
    
    if not os.path.exists(dataset_path):
    
        gnn_file, ref_image = prepare_data(scene_file, 
                                    max_depth = 5, 
                                    data_spp = 10, 
                                    ref_spp = 1000, 
                                    sensor = sensor, 
                                    output_folder = f'{main_folder}/predictions/generated/{model_name}')
        
        
        container = SimpleLightGraphContainer.fromfile(gnn_file, scene_file, ref_image, verbose=True)
            
        # build connections individually
        container.build_connections(n_graphs=10, n_nodes_per_graphs=5, n_neighbors=5, verbose=True)
        container = LightGraphManager.vstack(container)
        
        # prepare Dataset    
        data_list = []

        # TODO: improve this part
        for kid, (_, graphs) in enumerate(container.items()):
            
            for graph in graphs:
                torch_data = graph.data.to_torch()
                
                edge_attr = torch_data.edge_attr
                edge_attr[torch.isinf(torch_data.edge_attr)] = 0

                data = Data(x = torch_data.x, 
                        edge_index = torch_data.edge_index,
                        y = torch_data.y,
                        edge_attr = edge_attr,
                        pos = torch_data.pos)
                
                data_list.append(data)
            
            print(f'[Prepare torch data] progress: {(kid + 1) / len(container.keys()) * 100.:.2f}%', end='\r')
                
        # save dataset
        PathLightDataset(dataset_path, data_list)

    dataset = PathLightDataset(root=dataset_path)
   
    # normalize data
    model_folder = f'{main_folder}/models/{model_name}'
    
    x_scaler = skload(f'{model_folder}/x_node_scaler.bin')
    edge_scaler = skload(f'{model_folder}/x_edge_scaler.bin')
    # y_scaler = skload(f'{model_folder}/y_scaler.bin')
    
 
    scaled_dataset_path = f'{main_folder}/predictions/datasets/{output_name}_scaled'
    
    if encoder_enabled:
        print('[Encoded required] scaled data will be encoded')
        
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
 
    model = GNNL(hidden_channels=256, n_features=dataset.num_node_features)
    print(model)
    
    model.load_state_dict(torch.load(f'{model_folder}/model.pt'))
    model.eval()
            
    pixels = []
        
    n_predictions = len(dataset)
    for b_i in range(n_predictions):
        
        data = dataset[b_i]
        prediction = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)
        # prediction = y_scaler.inverse_transform(prediction.detach().numpy())
        pixels.append(prediction.detach().numpy())
        
        print(f'Prediction progress: {(b_i + 1) / len(dataset) * 100.:.2f}%', end='\r')
        
    image = np.array(pixels).reshape((h_size, w_size, 3))
    
    os.makedirs(f'{main_folder}/predictions', exist_ok=True)
    image_path = f'{main_folder}/predictions/{outfile_name}.exr'
    mi.util.write_bitmap(image_path, image)
    print(f'Predicted image has been saved into: {image_path}')
    
if __name__ == "__main__":
    main()
