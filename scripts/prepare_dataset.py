import os
import argparse
import random
import numpy as np
from itertools import chain

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

from mignn.dataset import PathLightDataset

import torch_geometric.transforms as GeoT

from joblib import dump as skdump
from joblib import load as skload

import tqdm
from multiprocessing.pool import ThreadPool

from utils import prepare_data, merge_by_chunk, init_normalizer
from utils import load_sensor_from, load_build_and_stack, scale_subset

from mignn.processing import ScalerTransform, SignalEncoder

import config as MIGNNConf

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
    dataset_percent   = MIGNNConf.DATASET_PERCENT
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

        # associate for each file in gnn_folder, the correct reference image
        gnn_files, references = list(zip(*list(chain.from_iterable(list([ 
                        [ (os.path.join(folder, g_file), ref_images[f_i]) for g_file in os.listdir(folder) ] 
                        for f_i, folder in enumerate(gnn_folders) 
                    ])))))
        
        print('\n[Building connections] creating connections using Mistuba3')
        # multiprocess build of connections
        pool_obj = ThreadPool()

        # load in parallel same scene file, imply error. Here we load multiple scenes
        params = list(zip(gnn_files,
                    [ scene_file for _ in range(len(gnn_files)) ],
                    [ output_temp for _ in range(len(gnn_files)) ],
                    references
                ))

        build_containers = []
        for result in tqdm.tqdm(pool_obj.imap(load_build_and_stack, params), total=len(params)):
            build_containers.append(result)

        # save intermediate PathLightDataset
        # Then fusion PathLightDatasets into only one
        # ensure file orders?
        intermediate_datasets_path = sorted(os.listdir(output_temp))
        random.shuffle(intermediate_datasets_path)
        
        x_scaler = init_normalizer(MIGNNConf.NORMALIZERS['x_node'])
        edge_scaler = init_normalizer(MIGNNConf.NORMALIZERS['x_edge'])
        y_scaler = init_normalizer(MIGNNConf.NORMALIZERS['y'])
    
        print(f'[Processing] fit scalers from {split_percent * 100}% of graphs (training set)')
        
        # prepare splitting of train and test dataset
        output_temp_train = f'{output_folder}/train/temp/train'
        os.makedirs(output_temp_train, exist_ok=True)
        
        output_temp_test = f'{output_folder}/train/temp/test'
        os.makedirs(output_temp_test, exist_ok=True)
    
        # compute scalers using partial fit and respecting train dataset
        n_graphs = 0
        n_train_graphs = 0
        
        train_data = []
        test_data = []
        
        partial_fit_normalizers = ['Standard', 'MinMax', 'LogStandard', 'LogMinMax']
        require_tracked_data = any([v not in partial_fit_normalizers + [None] for _, v in MIGNNConf.NORMALIZERS.items()])
        
        if require_tracked_data:
            print('[Warning] specified normalizers require the storage of all training data. \
                This can cause a memory surge.')
        
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
            indices = np.arange(n_elements)
            np.random.shuffle(indices)
            train_indices = indices[:split_index]
            
            # fill data
            for i in range(n_elements):
                
                # keep or not current data
                if random.random() <= dataset_percent:
                    
                    if i in train_indices:
                        train_data.append(subset[i]) 
                    else: 
                        test_data.append(subset[i])
                    
            # save intermediate dataset
            intermediate_train_dataset = PathLightDataset(temp_train_path, train_data)
            # only save
            PathLightDataset(temp_test_path, test_data, load=False) 
            
            # partial fit on train set when possible
            if MIGNNConf.NORMALIZERS['x_node'] in partial_fit_normalizers and x_scaler is not None:
                x_scaler.partial_fit(intermediate_train_dataset.data.x)
            
            if MIGNNConf.NORMALIZERS['x_edge'] in partial_fit_normalizers and edge_scaler is not None:
                edge_scaler.partial_fit(intermediate_train_dataset.data.edge_attr)
                
            if MIGNNConf.NORMALIZERS['y'] in partial_fit_normalizers and y_scaler is not None:
                y_scaler.partial_fit(intermediate_train_dataset.data.y.reshape(-1, 3))
                
            # reset train data list if necessary
            if not require_tracked_data:
                train_data = []
            
            # always clear test data
            test_data = []
            
        print(f'[Information] managed {n_graphs} graphs (train: {n_train_graphs}, test: {n_graphs - n_train_graphs}) ({dataset_percent*100:.2f}% of data (approximately) will be kept).')    
        
        # ensure normalization using scalers with no partial fit method
        if require_tracked_data:
            
            c_dataset, _ = PathLightDataset.collate(train_data)
            
            if MIGNNConf.NORMALIZERS['x_node'] not in partial_fit_normalizers and x_scaler is not None:
                x_scaler.fit(c_dataset.x)
            
            if MIGNNConf.NORMALIZERS['x_edge'] not in partial_fit_normalizers and edge_scaler is not None:
                edge_scaler.fit(c_dataset.edge_attr)
                
            if MIGNNConf.NORMALIZERS['y'] not in partial_fit_normalizers and y_scaler is not None:
                y_scaler.fit(c_dataset.y.reshape(-1, 3))
            
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
        transforms_list.append(SignalEncoder(MIGNNConf.ENCODING, MIGNNConf.MASK))
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
        
        # multi-process scale of dataset
        pool_obj_scaled = ThreadPool()
    
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
        
        # print(f'[Before merging] memory usage is: {psutil.virtual_memory().percent}%')
        
        # call merge by chunk for each scaled train and test subsets
        scaled_train_subsets = [ os.path.join(output_scaled_temp_train, p) \
                    for p in sorted(os.listdir(output_scaled_temp_train)) ]
        scaled_test_subsets = [ os.path.join(output_scaled_temp_test, p) \
            for p in sorted(os.listdir(output_scaled_temp_test)) ]
        
        # merged data into expected chunk size
        merge_by_chunk('train', scaled_train_subsets, train_dataset_path, applied_transforms)
        merge_by_chunk('test', scaled_test_subsets, test_dataset_path, applied_transforms)
                 
        print('[Cleaning] clear intermediates saved datasets...')
        os.system(f'rm -r {output_temp}') 
        os.system(f'rm -r {output_temp_scaled}') 
    else:
        print(f'[Information] {dataset_path} already generated')
 
if __name__ == "__main__":
    main()
