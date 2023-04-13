"""Module for utils functions (here for the moment)
"""
import os
import numpy as np
import sys
import math
import json

import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import subprocess

import mitsuba as mi
from mitsuba import ScalarTransform4f as T

from mignn.dataset import PathLightDataset

import config as MIGNNConf
      
def chunk_file(filename, output_folder, chunk_memory_size, sort_chunks=False):
    
    if MIGNNConf.SCENE_REVERSE:
        extract_key = lambda x: tuple(map(int, x.split(';')[0].split(',')))[::-1]
    else:
        extract_key = lambda x: tuple(map(int, x.split(';')[0].split(',')))
    
    with open(filename, 'r', encoding='utf-8') as f_gnn:
        rows = f_gnn.readlines()
        
        keys_and_rows = [ (extract_key(row), row) for row in rows ]
        pixels_rows = {}
        for k, row in keys_and_rows: 
            if k not in pixels_rows:
                pixels_rows[k] = []
            pixels_rows[k].append(row)
        
    # not need sort keys
    # chunk_rows = [list(res)[i: i + chunk_size] for i in range(0, len(res.keys()), chunk_size)]
    
    # TODO: if necessary to do conv, it is there
    # manage chunk using memory in Mo
    chunk_in_bytes = chunk_memory_size * (1024 ** 2)
    chunk_rows = []
    
    current_chunk = []
    memory_sum = 0
    
    # enable of not sorted keys (could be necessary when predicting)
    if sort_chunks:
        pixels_items = sorted(pixels_rows.items())
    else:
        pixels_items = pixels_rows.items()    
    
    # create chunks
    for key, rows in pixels_items:
        
        c_memory_bytes = sum(sys.getsizeof(line) for line in rows)
        memory_sum += c_memory_bytes
        
        if c_memory_bytes > chunk_in_bytes:
            raise ValueError(f'Cannot save information using only {chunk_memory_size} Mo \
                when generating GNN files')
            
        # create new chunk
        if memory_sum > chunk_in_bytes:
            chunk_rows.append(current_chunk)
            memory_sum = 0
            current_chunk = []
            
        current_chunk.append(key)
        
    # last save if needed
    if len(current_chunk) > 0:
        chunk_rows.append(current_chunk)
    
    # for i, chunk_keys in enumerate(chunks_dict(sorted(res), chunk_size)):
    for i, chunk_keys in enumerate(chunk_rows):
        
        _, folder_name = os.path.split(output_folder)
        
        # add specific index
        i_str = str(i)
        while len(i_str) < 5: 
            i_str = f'0{i_str}'
        
        filename = f'{folder_name}_{i_str}.path'
        with open(f'{os.path.join(output_folder, filename)}', 'w', encoding='utf-8') as f_output:
            
            # write row for current chunk keys
            for key in chunk_keys:
                pixel_rows = pixels_rows[key]
                for row in pixel_rows:
                    f_output.write(row)
            

def load_sensor_from(img_size, sensor_file):
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
        },
    })

def prepare_data(scene_file, max_depth, data_spp, ref_spp, sensors, output_folder, sort_chunks=False):
    """Enable to extract GNN data from `pathgnn` integrator and associated reference
    """
    os.makedirs(output_folder, exist_ok=True)

    scene = mi.load_file(scene_file)

    ref_integrator = mi.load_dict({'type': 'path', 'max_depth': max_depth})
    gnn_integrator = mi.load_dict({'type': 'pathgnn', 'max_depth': max_depth})

    # generate gnn file data and references
    ref_images = []
    low_images = []
    output_gnn_folders = []

    print(f'Generation of {len(sensors)} views for `{scene_file}`')
    for view_i, sensor in enumerate(sensors):

        ref_image_path = f'{output_folder}/ref_{view_i}.exr'

        if not os.path.exists(ref_image_path):
            # print(f'Generating data for view n°{view_i+1}')
            ref_image = mi.render(scene, spp=ref_spp, integrator=ref_integrator, sensor=sensor)

            # save image as exr and reload it using cv2
            cv2.imwrite(ref_image_path, np.asarray(ref_image))

        ref_images.append(ref_image_path)

        # print(f' -- reference of view n°{view_i+1} generated...')
        params = mi.traverse(scene)
        gnn_log_filename = f'{output_folder}/gnn_file_{view_i}.path'
        low_image_path = f'{output_folder}/low_{view_i}.exr'
        params['logfile'] = gnn_log_filename
        params.update();

        # split data into multiple files (better to later load data)
        gnn_log_folder = f'{output_folder}/gnn_file_{view_i}'
        
        if not os.path.exists(gnn_log_folder):
            low_image = mi.render(scene, spp=data_spp, integrator=gnn_integrator, sensor=sensor)
    
            cv2.imwrite(low_image_path, np.asarray(low_image))
        
            # now split file into multiple ones (except when predict)
            os.makedirs(gnn_log_folder, exist_ok=True)
        
            # need to chunk file by pixels keys
            chunk_file(gnn_log_filename, gnn_log_folder, MIGNNConf.VIEWPOINT_CHUNK, sort_chunks)
            
            # now remove initial log file
            os.system(f'rm {gnn_log_filename}')
                
        print(f'[Data generation] GNN data progress: {(view_i+1) / len(sensors) * 100:.2f}%', end='\r')

        low_images.append(low_image_path)
        output_gnn_folders.append(gnn_log_folder)

    return output_gnn_folders, ref_images, low_images


def load_build_and_stack(params):

    gnn_file, scene_file, output_temp, ref_image_path = params

    # [Important] this task cannot be done by multiprocess, need to be done externaly
    # Mitsuba seems to be concurrent package inside same context program

    process = subprocess.Popen(["python", "load_build_and_stack.py", \
        "--gnn_file", gnn_file, \
        "--scene", scene_file, \
        "--reference", ref_image_path, \
        "--output", output_temp])
    process.wait()

    # build_container = dill.load(open(expected_container_path, 'rb'))

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

def init_normalizer(normalizer_name):
    """Get the expected sklearn normalizer
    """
    
    if normalizer_name == 'MinMax':
        return MinMaxScaler()

    if normalizer_name == 'Norm':
        return Normalizer()
    
    if normalizer_name == 'Standard':
        return StandardScaler()
    
    return None