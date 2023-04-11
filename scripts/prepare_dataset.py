import os
import uuid
import argparse
import psutil
import numpy as np
import math
import json

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

from mignn.dataset import PathLightDataset

import torch_geometric.transforms as GeoT

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from joblib import dump as skdump
from joblib import load as skload

import tqdm
from multiprocessing.pool import ThreadPool

from utils import prepare_data
from utils import load_sensor_from, load_build_and_stack, scale_subset

from transforms import ScalerTransform, SignalEncoder

import config as MIGNNConf


def merge_by_chunk(scaled_datasets_path, output_path, applied_transforms):
    
    memory_sum = 0
    memory_size_in_bytes = MIGNNConf.DATASET_CHUNK * (1024 ** 2)
    
    print(f'[Before merging] memory usage is: {psutil.virtual_memory().percent}%')
    
    data_list = []
    
    n_subsets = len(scaled_datasets_path)
    step = (n_subsets // 100) + 1
    
    # also store metadata file
    n_batchs = 0
    n_samples = 0
    n_node_features = None
    n_target_features = None
    
    for idx, scaled_dataset_path in enumerate(scaled_datasets_path):
        
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
                
                print(f'[During merging] memory usage is: {psutil.virtual_memory().percent}%')
        
                # get the expected dataset folder
                c_dataset_path = os.path.join(output_path, f'{str(uuid.uuid4())}.path')
                
                # save intermediate dataset with expected max size
                PathLightDataset(c_dataset_path, data_list, load=False)
                
                # reset data list
                data_list = []
                
                # reset memory sum
                memory_sum = 0
                
                
            data_list.append(data)
            n_samples += 1
                        
        # clear memory
        del c_scaled_dataset
        
        if (idx % step == 0 or idx >= n_subsets - 1):
            print(f'[Prepare dataset (chunks of: {MIGNNConf.DATASET_CHUNK} Mo)] -- progress: {(idx + 1) / n_subsets * 100.:.0f}%', \
                end='\r' if idx + 1 < n_subsets else '\n')
        
    # do last save if needed    
    if len(data_list) > 0:
        n_batchs += math.ceil(len(data_list) / MIGNNConf.BATCH_SIZE)
                
        c_dataset_path = os.path.join(output_path, f'{str(uuid.uuid4())}.path')
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


def main():

    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--output', type=str, help="output dataset folder", required=True)
    parser.add_argument('--sensors', type=str, help="file with all viewpoints on scene", required=True)
    
    args = parser.parse_args()

    scene_file        = args.scene
    output_folder     = args.output
    sensors_folder    = args.sensors

    # Some MIGNN params 
    split_percent     = MIGNNConf.TRAINING_SPLIT
    w_size, h_size    = MIGNNConf.VIEWPOINT_SIZE
    sensors_n_samples = MIGNNConf.VIEWPOINT_SAMPLES

    # use of: https://github.com/prise-3d/vpbrt
    # read from camera LookAt folder
    sensors = []
    for file in sorted(os.listdir(sensors_folder)):
        file_path = os.path.join(sensors_folder, file)

        sensor = load_sensor_from((w_size, h_size), file_path)

        # Use a number of times the same sensors in order to increase knowledge
        # Multiple GNN files will be generated
        for _ in range(sensors_n_samples):
            sensors.append(sensor)

    os.makedirs(output_folder, exist_ok=True)
    dataset_path = f'{output_folder}/train/datasets'

    model_folder = f'{output_folder}/model'
    scalers_folder = f'{output_folder}/model/scalers'
    os.makedirs(model_folder, exist_ok=True)
    
    # multiple datasets to avoid memory overhead
    output_temp = f'{output_folder}/train/temp/'
    os.makedirs(output_temp, exist_ok=True)
    
    # same for scaled datasets
    output_temp_scaled = f'{output_folder}/train/temp_scaled/'
    os.makedirs(output_temp_scaled, exist_ok=True)

    if not os.path.exists(scalers_folder):
        print('[Data generation] start generating GNN data using Mistuba3')
        gnn_folders, ref_images, _ = prepare_data(scene_file,
                                    max_depth = MIGNNConf.MAX_DEPTH,
                                    data_spp = MIGNNConf.GNN_SPP,
                                    ref_spp = MIGNNConf.REF_SPP,
                                    sensors = sensors,
                                    output_folder = f'{output_folder}/train/generated')

        print('\n[Building connections] creating connections using Mistuba3')
        # multiprocess build of connections
        pool_obj = ThreadPool()

        # load in parallel same scene file, imply error. Here we load multiple scenes
        params = list(zip(gnn_folders,
                    [ scene_file for _ in range(len(gnn_folders)) ],
                    [ output_temp for _ in range(len(gnn_folders)) ],
                    ref_images
                ))

        build_containers = []
        for result in tqdm.tqdm(pool_obj.imap(load_build_and_stack, params), total=len(params)):
            build_containers.append(result)

        # save intermediate PathLightDataset
        # Then fusion PathLightDatasets into only one
        # ensure file orders
        intermediate_datasets_path = sorted(os.listdir(output_temp))
        
        x_scaler = MinMaxScaler()
        edge_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    
        print(f'[Processing] fit scalers from {split_percent * 100}% of graphs (training set)')
        
        # prepare splitting of train and test dataset
        output_temp_train = f'{output_folder}/train/temp/train'
        os.makedirs(output_temp_train, exist_ok=True)
        
        output_temp_test = f'{output_folder}/train/temp/test'
        os.makedirs(output_temp_test, exist_ok=True)
    
        # compute scalers using partial fit and respecting train dataset
        n_graphs = 0
        n_train_graphs = 0
        
        for dataset_name in intermediate_datasets_path:
            
            c_dataset_path = os.path.join(output_temp, dataset_name)
            subset = PathLightDataset(root=c_dataset_path)
            
            # record data
            n_elements = len(subset)
            n_graphs += n_elements
            
            # split data into training and testing set
            split_index = int(n_elements * split_percent)
            n_train_graphs += split_index
            
            # create train and test datasets
            temp_train_path = os.path.join(output_temp_train, dataset_name)
            temp_test_path = os.path.join(output_temp_test, dataset_name)
                    
            # shuffle data
            indices = np.arange(split_index)
            np.random.shuffle(indices)
            train_indices = indices[:split_index]
            
            train_data = []
            test_data = []
            
            # fill data
            for i in range(n_elements):
                if i in train_indices:
                    train_data.append(subset[i]) 
                else: 
                    test_data.append(subset[i])
                    
            # save intermediate dataset
            intermediate_train_dataset = PathLightDataset(temp_train_path, train_data)
            # only save
            PathLightDataset(temp_test_path, test_data) 
            
            # partial fit on test set
            x_scaler.partial_fit(intermediate_train_dataset.data.x)
            edge_scaler.partial_fit(intermediate_train_dataset.data.edge_attr)
            y_scaler.partial_fit(intermediate_train_dataset.data.y.reshape(-1, 3))
            
        
        print(f'[Information] dataset is composed of {n_graphs} graphs (train: {n_train_graphs}, test: {n_graphs - n_train_graphs})')    
        
        # save scalers
        os.makedirs(scalers_folder, exist_ok=True)
        
        skdump(x_scaler, f'{scalers_folder}/x_node_scaler.bin', compress=True)
        skdump(edge_scaler, f'{scalers_folder}/x_edge_scaler.bin', compress=True)
        skdump(y_scaler, f'{scalers_folder}/y_scaler.bin', compress=True)


    x_scaler = skload(f'{scalers_folder}/x_node_scaler.bin')
    edge_scaler = skload(f'{scalers_folder}/x_edge_scaler.bin')
    y_scaler = skload(f'{scalers_folder}/y_scaler.bin')
        
    # reload scalers    
    scalers = {
        'x_node': x_scaler,
        'x_edge': edge_scaler,
        'y': y_scaler
    }
    
    transforms_list = [ScalerTransform(scalers)]
    
    if MIGNNConf.ENCODING is not None:
        print('[Scaling (with encoding)] start preparing encoded scaled data...')
        transforms_list.append(SignalEncoder(MIGNNConf.ENCODING))
    else:
        print('[Scaling] start preparing scaled data...')
    applied_transforms = GeoT.Compose(transforms_list)    

    # applied transformations over all intermediate path light dataset
    # avoid memory overhead
    if not os.path.exists(f'{dataset_path}_train'):
        
        output_scaled_temp_train = f'{output_folder}/train/temp_scaled/train'
        os.makedirs(output_scaled_temp_train, exist_ok=True)
        
        output_scaled_temp_test = f'{output_folder}/train/temp_scaled/test'
        os.makedirs(output_scaled_temp_test, exist_ok=True)
        
        # concat train and test intermediate datasets with expected output
        intermediate_datasets_path = [ (output_scaled_temp_train, os.path.join(output_temp_train, p)) \
                                    for p in sorted(os.listdir(output_temp_train)) ]
        intermediate_datasets_path += [ (output_scaled_temp_test, os.path.join(output_temp_test, p)) \
                                    for p in sorted(os.listdir(output_temp_test)) ]
        
        # separate expected output and where to intermediate dataset path
        output_scaled, datasets_path = list(zip(*intermediate_datasets_path))
        
        n_intermediates = len(intermediate_datasets_path)
        intermediate_scaled_datasets_path = []
        
        print(f'[Before intermediate] memory usage is: {psutil.virtual_memory().percent}%')
        
        # multi-process scale of dataset
        pool_obj_scaled = ThreadPool()
    
        # load in parallel same scene file, imply error. Here we load multiple scenes
        scaled_params = list(zip(datasets_path,
                    [ scalers_folder for _ in range(n_intermediates) ],
                    output_scaled
                ))
        
        for result in tqdm.tqdm(pool_obj_scaled.imap(scale_subset, scaled_params), total=len(scaled_params)):
            intermediate_scaled_datasets_path.append(result)
            
        # create output train and test folders
        train_dataset_path = f'{dataset_path}_train'
        test_dataset_path = f'{dataset_path}_test'
        os.makedirs(train_dataset_path, exist_ok=True)
        os.makedirs(test_dataset_path, exist_ok=True)
        
        print(f'[Before merging] memory usage is: {psutil.virtual_memory().percent}%')
        
        # call merge by chunk for each scaled train and test subsets
        scaled_train_subsets = [ os.path.join(output_scaled_temp_train, p) \
                    for p in sorted(os.listdir(output_scaled_temp_train)) ]
        scaled_test_subsets = [ os.path.join(output_scaled_temp_test, p) \
            for p in sorted(os.listdir(output_scaled_temp_test)) ]
        
        # merged data into expected chunk size
        merge_by_chunk(scaled_train_subsets, train_dataset_path, applied_transforms)
        merge_by_chunk(scaled_test_subsets, test_dataset_path, applied_transforms)
                 
        print(f'[After merging] memory usage is: {psutil.virtual_memory().percent}%')
        print('[Cleaning] clear intermediates saved datasets...')
        os.system(f'rm -r {output_temp}') 
        os.system(f'rm -r {output_temp_scaled}') 
    else:
        print(f'[Information] {dataset_path} already generated')
 
if __name__ == "__main__":
    main()
