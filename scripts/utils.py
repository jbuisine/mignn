"""Module for utils functions (here for the moment)
"""
import os
import numpy as np
import math
import json
import msgpack
import torch
from torch_geometric.data import Data

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import subprocess

import mitsuba as mi
from mitsuba import ScalarTransform4f as T

from mignn.dataset import PathLightDataset

import config as MIGNNConf
      
def load_and_convert(filename):
    #global done
    graphs = []
    with open(filename, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
        
        # for each key data, extract graph
        for key, k_data in data.items():
            
            pixel = list(map(int, key.split(',')))
            
            # nodes data
            x_node = torch.tensor(k_data["x"], dtype=torch.float)
            x_node_pos = torch.tensor(k_data["pos"], dtype=torch.float)
            x_node_primary = torch.tensor(k_data["x_primary"], dtype=torch.bool)
            
            # edges data
            edge_index = torch.tensor(k_data["edge_index"], dtype=torch.long)    
            edge_attr = torch.tensor(k_data["edge_attr"], dtype=torch.float)
            edge_built = torch.tensor(k_data["edge_built"], dtype=torch.bool)
            
            # targets
            y_targets = torch.tensor(k_data["y"], dtype=torch.float)
            
            graph_data = Data(x=x_node, 
                            x_primary=x_node_primary, 
                            pos=x_node_pos,
                            edge_index=edge_index.t().contiguous(), 
                            edge_attr=edge_attr,
                            edge_built=edge_built, 
                            y=y_targets, 
                            pixel=pixel)
            graphs.append(graph_data)
    
    return graphs

def load_sensor_from(img_size, sensor_file, integrator, gnn_until, gnn_nodes, gnn_neighbors):
    """Build a new mitsuba sensor from perspective camera information
    """

    with open(sensor_file, 'r', encoding='utf-8') as f:
        look_at_data = f.readline().replace('\n', '').split('  ')
        fov = float(f.readline().replace('\n', '').split(' ')[-1])
        origin = list(map(float, look_at_data[1].split(' ')))
        target = list(map(float, look_at_data[2].split(' ')))
        up = list(map(float, look_at_data[3].split(' ')))
        # print(f'LookAt: [origin: {origin}, target: {target}, up: {up}], Fov: {fov}')

    w_size, h_size = img_size

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
            'gnn_integrator_type': integrator,
            'gnn_until': gnn_until,
            'gnn_nodes': gnn_nodes,
            'gnn_neighbors': gnn_neighbors
        },
    })

def prepare_data(scene_file, integrator, max_depth, ref_spp, sensors, output_folder):
    """Enable to extract GNN data from `pathgnn` integrator and associated reference
    """
    os.makedirs(output_folder, exist_ok=True)

    scene = mi.load_file(scene_file)

    gnn_integrator = mi.load_dict({'type': integrator, 'max_depth': max_depth})

    # generate gnn file data and references
    output_gnn_folders = []

    print(f'Generation of {len(sensors)} views for `{scene_file}`')
    for view_i, sensor in enumerate(sensors):

        gnn_log_folder = f'{output_folder}/gnn_folder_{view_i}'
        
        params = mi.traverse(scene)
        params['output_gnn'] = gnn_log_folder
        params.update();
        
        if not os.path.exists(gnn_log_folder):
            # print(f'Generating data for view n°{view_i+1}')
            mi.render(scene, spp=ref_spp, integrator=gnn_integrator, sensor=sensor)

        # print(f' -- reference of view n°{view_i+1} generated...')
  
        print(f'[Data generation] GNN data progress: {(view_i+1) / len(sensors) * 100:.2f}%', end='\r')

        output_gnn_folders.append(gnn_log_folder)

    return output_gnn_folders


def load_and_save(params):

    gnn_file, output_temp = params

    # [Important] this task cannot be done by multiprocess, need to be done externaly
    # Mitsuba seems to be concurrent package inside same context program

    process = subprocess.Popen(["python", "load_and_save.py", \
        "--gnn_file", gnn_file, \
        "--output", output_temp])
    process.wait()

    return True


def scale_subset(params):

    dataset_path, scalers_path, output_temp_scaled = params

    # [Important] this task cannot be done by multiprocess, need to be done externaly
    process = subprocess.Popen(["python", "scale_subset.py", \
        "--dataset", dataset_path, \
        "--scalers", scalers_path, \
        "--output", output_temp_scaled])
    process.wait()
    
    _, dataset_name = os.path.split(dataset_path)

    return os.path.join(output_temp_scaled, dataset_name)

def merge_by_chunk(output_name, scaled_datasets_path, output_path, applied_transforms):
    
    memory_sum = 0
    memory_size_in_bytes = MIGNNConf.DATASET_CHUNK * (1024 ** 2)
    
    data_list = []
    
    n_subsets = len(scaled_datasets_path)
    step = (n_subsets // 100) + 1
    
    # also store metadata file
    n_batchs = 0
    n_samples = 0
    n_node_features = None
    n_target_features = None
    n_saved = 0
    for idx, scaled_dataset_path in enumerate(sorted(scaled_datasets_path)):
        
        # TODO: check if really required to get pre_transform param
        c_scaled_dataset = PathLightDataset(root=scaled_dataset_path, 
                                        pre_transform=applied_transforms)
        
        n_current_samples = len(c_scaled_dataset)
        
        if n_node_features is None:
            n_node_features = c_scaled_dataset.num_node_features
            n_target_features = c_scaled_dataset.num_target_features
        
        for c_data_i in range(n_current_samples):
            data = c_scaled_dataset[c_data_i]
            
            # get current data memory size
            memory_object = sum([v.element_size() * v.numel() for k, v in data])
            memory_sum += memory_object
                    
            # need to store into intermediate dataset
            # if limited memory is greater than fixed or end of train dataset size
            if memory_sum > memory_size_in_bytes:
        
                n_batchs += math.ceil(len(data_list) / MIGNNConf.BATCH_SIZE)
                
                # save using specific index
                i_str = str(n_saved)
                while len(i_str) < 5: 
                    i_str = f'0{i_str}'
                    
                c_dataset_path = os.path.join(output_path, f'merged_scaled_{i_str}.path')
                
                # save intermediate dataset with expected max size
                PathLightDataset(c_dataset_path, data_list, load=False)
                
                # reset data list
                data_list = []
                
                # reset memory sum
                memory_sum = 0
                n_saved += 1
                
                
            data_list.append(data)
            n_samples += 1
                        
        # clear memory
        del c_scaled_dataset
        
        if (idx % step == 0 or idx >= n_subsets - 1):
            print(f'[Prepare {output_name} dataset (with chunks of: {MIGNNConf.DATASET_CHUNK} Mo)] -- progress: {(idx + 1) / n_subsets * 100.:.0f}%', \
                end='\r' if idx + 1 < n_subsets else '\n')
        
    # do last save if needed    
    if len(data_list) > 0:
        
        n_saved += 1
        
        i_str = str(n_saved)
        while len(i_str) < 5: 
            i_str = f'0{i_str}'
            
        n_batchs += math.ceil(len(data_list) / MIGNNConf.BATCH_SIZE)
                
        c_dataset_path = os.path.join(output_path, f'merged_scaled_{i_str}.path')
        
        # save intermediate dataset with expected max size
        PathLightDataset(c_dataset_path, data_list, load=False)
        
    # save training metadata
    metadata = { 
        'n_samples': n_samples, 
        'n_batchs': n_batchs,
        'n_node_features': n_node_features,
        'n_target_features': n_target_features,
    }
    
    with open(f'{output_path}/metadata', 'w', encoding='utf-8') as outfile:
        json.dump(metadata, outfile)


def init_loss(loss_name):
    """Get the expected torch loss
    """
    
    if loss_name == 'MSE':
        return torch.nn.MSELoss()

    if loss_name == 'MAE':
        return torch.nn.L1Loss()
    
    if loss_name == 'Huber':
        return torch.nn.HuberLoss()
    
    return None