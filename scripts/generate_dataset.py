import os
import argparse
import shutil
import random
import numpy as np
from itertools import chain

from mignn.dataset import PathLightDataset

import torch_geometric.transforms as GeoT

from joblib import dump as skdump
from joblib import load as skload

import tqdm
from multiprocessing.pool import ThreadPool

from utils import scale_viewpoint_and_merge

from mignn.processing import ScalerTransform, SignalEncoder
from mignn.processing.scalers import ScalersManager

import config as MIGNNConf

def main():

    parser = argparse.ArgumentParser(description="Scaled data from config using GNN data")
    parser.add_argument('--data', type=str, help="GNN data", required=True)
    parser.add_argument('--output', type=str, help="output dataset folder", required=True)
    
    args = parser.parse_args()

    output_folder     = args.output
    input_containers  = args.data

    # Some MIGNN params 
    dataset_percent   = MIGNNConf.DATASET_PERCENT

    os.makedirs(output_folder, exist_ok=True)
    dataset_path = f'{output_folder}/datasets/data'

    scalers_folder = f'{output_folder}/datasets/scalers'
    
    # intermediates datasets when fitting scalers
    output_temp_train = f'{output_folder}/datasets/intermediates/train'
    output_temp_test = f'{output_folder}/datasets/intermediates/test'
    
    # same for scaled datasets
    output_temp_scaled = f'{output_folder}/datasets/intermediates/scaled'
    os.makedirs(output_temp_scaled, exist_ok=True)

    if not os.path.exists(scalers_folder):
                
        # clear previous possible scaled data folders
        if os.path.exists(output_temp_train):
            shutil.rmtree(output_temp_train)
        
        if os.path.exists(output_temp_test):
            shutil.rmtree(output_temp_test)

        # save intermediate PathLightDataset
        # Then fusion PathLightDatasets into only one
        train_viewpoints = os.path.join(input_containers, 'train')
        test_viewpoints = os.path.join(input_containers, 'test')
        
        # initialize scalers from config using manager 
        scalers = ScalersManager(config=MIGNNConf.NORMALIZERS)
        
        train_viewpoints_folder = [ os.path.join(train_viewpoints, v_name) for v_name in os.listdir(train_viewpoints) ]
        test_viewpoints_folder = [ os.path.join(test_viewpoints, v_name) for v_name in os.listdir(test_viewpoints) ]
        
        n_train_viewpoints = len(train_viewpoints_folder)
        print(f'[Processing] fit scalers on {n_train_viewpoints} viewpoints')
        
        if not scalers.enable_partial:
            raise AttributeError('[Unsupported] specified normalizers require the storage of all training data. \
                This can cause a memory surge.')
            
        # retrieve all containers filename
        train_viewpoints_files = list(chain(*[
                [ os.path.join(viewpoint_path, c_file) for c_file in os.listdir(viewpoint_path) ]
                for viewpoint_path in train_viewpoints_folder
            ]))
        
        n_graphs = 0
        
        for idx, c_dataset_path in enumerate(train_viewpoints_files):
            
            subset = PathLightDataset(root=c_dataset_path)

            # record data
            n_graphs += len(subset)
            
            # partial fit on train set when possible
            scalers.partial_fit(subset)
                
            # # reset train data list if necessary
            # if scalers.enable_partial:
            #     train_data = []
            
            # # always clear test data
            # test_data = []
            
            print(f'[Processing] fit scalers progress: {(idx + 1) / len(train_viewpoints_files) * 100.:.2f}%', end='\r')
                
        print(f'[Information] {n_graphs} graphs have been managed.')    
        
        # For the moment we avoid total fit scalers and raise issue
        # ensure normalization using scalers with no partial fit method
        # if not scalers.enable_partial:
            
        #     c_dataset, _ = PathLightDataset.collate(train_data)
        #     scalers.fit(c_dataset)
            
        # save scalers
        os.makedirs(scalers_folder, exist_ok=True)
        
        skdump(scalers, f'{scalers_folder}/scalers.bin', compress=True)

    # reload scalers 
    scalers_path = f'{scalers_folder}/scalers.bin'   
    print('[Scaling] start preparing scaled data...') 

    # applied transformations over all intermediate path light dataset
    # avoid memory overhead
    if not os.path.exists(f'{dataset_path}/train'):
        
        output_scaled_temp_train = f'{output_temp_scaled}/train'
        output_scaled_temp_test = f'{output_temp_scaled}/test'
        os.makedirs(output_scaled_temp_train, exist_ok=True)
        os.makedirs(output_scaled_temp_test, exist_ok=True)
        
        train_dataset_path = f'{dataset_path}/train'
        test_dataset_path = f'{dataset_path}/test'
        os.makedirs(train_dataset_path, exist_ok=True)
        os.makedirs(test_dataset_path, exist_ok=True)
        
        # create params (train and test)
        viewpoints_params = list([
                (
                    viewpoint_path,
                    scalers_path,
                    os.path.join(output_scaled_temp_train, os.path.split(viewpoint_path)[-1]),
                    os.path.join(train_dataset_path, os.path.split(viewpoint_path)[-1])
                ) 
            for viewpoint_path in train_viewpoints_folder
        ])
        
        viewpoints_params += list([
                (
                    viewpoint_path,
                    scalers_path,
                    os.path.join(output_scaled_temp_test, os.path.split(viewpoint_path)[-1]),
                    os.path.join(test_dataset_path, os.path.split(viewpoint_path)[-1])
                ) 
            for viewpoint_path in test_viewpoints_folder
        ])
    
        # multi-process scale of dataset
        pool_obj_scaled = ThreadPool()
        pool_results = []
    
        for result in tqdm.tqdm(pool_obj_scaled.imap(scale_viewpoint_and_merge, viewpoints_params), total=len(viewpoints_params)):
            pool_results.append(result)
    else:
        print(f'[Information] {dataset_path} already generated')
 
if __name__ == "__main__":
    main()
