"""Module for utils functions (here for the moment)
"""
import os
import numpy as np
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import subprocess
from collections import defaultdict

import mitsuba as mi
from mitsuba import ScalarTransform4f as T

from torch_geometric.data import Data

from mignn.processing.encoder import signal_encoder

from config import CHUNK_SIZE

from itertools import islice

def chunks_dict(data, size=10000):
    
    iterator = iter(data)
    for _ in range(0, len(data), size):
        yield {k:data[k] for k in islice(iterator, size)}
        
def chunk_file(filename, output_folder, chunk_size):
    
    extract_key = lambda x: list(map(int, x.split(';')[0].split(',')))
    
    with open(filename, 'r', encoding='utf-8') as f_gnn:
        rows = f_gnn.readlines()
        
        keys_and_rows = [ (extract_key(row), row) for row in rows ]
        res = defaultdict(list)
        for k, v in keys_and_rows: 
            res[tuple(k)].append(v)
            
        # sorted_rows = sorted(rows, key=extract_key, reverse=False)
        
    # chunk_rows = [list(res.keys())[i: i + n_chunks] for i in range(0, len(res.keys()), n_chunks)]
    # print(len(chunk_rows))
    
    for i, chunk_keys in enumerate(chunks_dict(res, chunk_size)):
        # print(item)
    # for i, chunk_keys in enumerate(chunk_rows):
        _, folder_name = os.path.split(output_folder)
        
        filename = f'{folder_name}_{i}.path'
        with open(f'{os.path.join(output_folder, filename)}', 'w', encoding='utf-8') as f_output:
            
            # write row for current chunk keys
            for key in chunk_keys:
                keys_rows = res[key]
                for row in keys_rows:
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

def prepare_data(scene_file, max_depth, data_spp, ref_spp, sensors, output_folder):
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
        
            # now split file into multiple ones
            os.makedirs(gnn_log_folder, exist_ok=True)
            
            # need to chunk file by pixels keys
            chunk_file(gnn_log_filename, gnn_log_folder, CHUNK_SIZE)
            
            # now remove initial log file
            os.system(f'rm {gnn_log_filename}')
            
        print(f'[Generation from images] GNN data progress: {(view_i+1) / len(sensors) * 100:.2f}%', end='\r')

        low_images.append(low_image_path)
        output_gnn_folders.append(gnn_log_folder)

    return output_gnn_folders, ref_images, low_images


def scale_data(dataset, scalers, encoding=False, encoder_size=None):

    scaled_data_list = []

    x_scaler = scalers['x_node']
    edge_scaler = scalers['x_edge']
    y_scaler = scalers['y']

    n_graphs = len(dataset)
    for d_i in range(n_graphs):

        data = dataset[d_i]

        # perform scale and then encoding
        x_data = torch.tensor(x_scaler.transform(data.x), dtype=torch.float)
        x_edge_data = torch.tensor(edge_scaler.transform(data.edge_attr), dtype=torch.float)
        y_data = torch.tensor(y_scaler.transform(data.y.reshape(-1, 3)), dtype=torch.float)

        if encoding:
            x_data = signal_encoder(x_data, L=encoder_size)
            x_edge_data = signal_encoder(x_edge_data, L=encoder_size)

        scaled_data = Data(x = x_data,
                edge_index = data.edge_index,
                y = y_data,
                edge_attr = x_edge_data,
                pos = data.pos)

        scaled_data_list.append(scaled_data)

        print(f' -- [Prepare scaled torch data] progress: {(d_i + 1) / n_graphs * 100.:.2f}%', end='\r')

    return scaled_data_list



def load_build_and_stack(params):

    gnn_folder, scene_file, output_temp, ref_image_path = params

    # [Important] this task cannot be done by multiprocess, need to be done externaly
    # Mitsuba seems to be concurrent package inside same context program

    process = subprocess.Popen(["python", "load_build_and_stack.py", \
        "--gnn_folder", gnn_folder, \
        "--scene", scene_file, \
        "--reference", ref_image_path, \
        "--output", output_temp])
    process.wait()

    # build_container = dill.load(open(expected_container_path, 'rb'))

    return True
