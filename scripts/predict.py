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
from mignn.processing.embedding import signal_embedding

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from joblib import load as skload

from models.gcn_model import GNNL

w_size, h_size = 128, 128

def load_sensor(r, phi, theta, target):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 25,
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
    parser.add_argument('--model', type=str, help="where to find model", required=False, default=10)
    parser.add_argument('--output', type=str, help="output prediction folder", required=True)
    parser.add_argument('--embedding', type=bool, help="embedding data or not", required=False, default=False)
    
    args = parser.parse_args()
    
    scene_file        = args.scene
    model_folder      = args.model
    output_name       = args.output
    embedding_enabled = args.embedding

    sensor_count = 6
    radius = 5
    phis = [ 140 - (i*20) for i in range(sensor_count)]
    theta = 22

    sensors = [load_sensor(radius, phi, theta, [0, 1, 0]) for phi in phis]
    current_sensor = sensors[2]
    
    
    dataset_path = f'data/predictions/datasets/{output_name}'
    
    if not os.path.exists(dataset_path):
    
        gnn_file, ref_image = prepare_data(scene_file, 
                                    max_depth = 5, 
                                    data_spp = 10, 
                                    ref_spp = 1000, 
                                    sensor = current_sensor, 
                                    output_folder = f'data/predictions/generated/{output_name}')
        
        
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
                
                # do embedding
                if embedding_enabled:
                    data = Data(x = signal_embedding(torch_data.x), 
                            edge_index = torch_data.edge_index,
                            y = torch_data.y,
                            edge_attr = signal_embedding(torch_data.edge_attr),
                            pos = torch_data.pos)
                else:
                    
                    edge_attr = torch_data.edge_attr
                    edge_attr[np.isinf(torch_data.edge_attr)] = 0

                    data = Data(x = torch_data.x, 
                            edge_index = torch_data.edge_index,
                            y = torch_data.y,
                            edge_attr = edge_attr,
                            pos = torch_data.pos)
                    
                data_list.append(data)
            
            print(f'[Prepare torch data] progress: {(kid + 1) / len(container.keys()) * 100.:.2f}%', end='\r')
                
        # save dataset
        PathLightDataset(dataset_path, data_list)

    # transform applied only when loaded
    dataset = PathLightDataset(root=dataset_path)
    print(dataset[0])
    
    # normalize data
    x_scaler = skload(f'{model_folder}/x_node_scaler.bin')
    edge_scaler = skload(f'{model_folder}/x_edge_scaler.bin')
    y_scaler = skload(f'{model_folder}/y_scaler.bin')
    
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = GNNL(hidden_channels=128, n_features=dataset.num_node_features)
    print(model)
    
    model.load_state_dict(torch.load(f'{model_folder}/model.pt'))
    model.eval()
            
    pixels = []
        
    for b_i in range(len(dataset)):
        
        data = dataset[b_i]
        batch = torch.zeros(len(data.x), dtype=torch.int64)
        x_data = torch.tensor(x_scaler.transform(data.x), dtype=torch.float)
        x_edge_data = torch.tensor(edge_scaler.transform(data.edge_attr), dtype=torch.float)
        
        prediction = model(x_data, x_edge_data, data.edge_index, batch=batch)
        prediction = y_scaler.inverse_transform(prediction.detach().numpy())
        pixels.append(prediction)
        
        print(f'Prediction progress: {(b_i + 1) / len(dataset) * 100.:.2f}%', end='\r')
        
    image = np.array(pixels).reshape((h_size, w_size, 3))
    
    os.makedirs('data/predictions', exist_ok=True)
    image_path = f'data/predictions/{output_name}.exr'
    mi.util.write_bitmap(image_path, image)
    print(f'Predicted image has been saved into: {image_path}')
    
if __name__ == "__main__":
    main()
