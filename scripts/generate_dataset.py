import os
import argparse
import shutil
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

from utils import prepare_data, merge_by_chunk
from utils import load_sensor_from, load_build_and_stack, scale_subset

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
    split_percent     = MIGNNConf.TRAINING_SPLIT
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
        # ensure file orders?
        intermediate_datasets_path = sorted(os.listdir(input_containers))
        random.shuffle(intermediate_datasets_path)
        
        # initialize scalers from config using manager 
        scalers = ScalersManager(config=MIGNNConf.NORMALIZERS)
        
        print(f'[Processing] fit scalers from {split_percent * 100}% of graphs (training set)')
        
        # prepare splitting of train and test dataset
        os.makedirs(output_temp_train, exist_ok=True)
        os.makedirs(output_temp_test, exist_ok=True)

        # compute scalers using partial fit and respecting train dataset
        n_graphs = 0
        n_train_graphs = 0
        
        train_data = []
        test_data = []
        
        if not scalers.enable_partial:
            raise AttributeError('[Unsupported] specified normalizers require the storage of all training data. \
                This can cause a memory surge.')
        
        n_intermediates_unscaled = len(intermediate_datasets_path)
        
        for idx, dataset_name in enumerate(intermediate_datasets_path):
                
            c_dataset_path = os.path.join(input_containers, dataset_name)
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
            scalers.partial_fit(intermediate_train_dataset)
                
            # reset train data list if necessary
            if scalers.enable_partial:
                train_data = []
            
            # always clear test data
            test_data = []
            
            print(f'[Processing] fit scalers progress: {(idx + 1) / n_intermediates_unscaled * 100.:.2f}%', end='\r')
            
        print(f'[Information] managed {n_graphs} graphs (train: {int(n_train_graphs * dataset_percent)}, test: {(n_graphs - n_train_graphs) * dataset_percent}).')    
        
        # For the moment we avoid total fit scalers and raise issue
        # ensure normalization using scalers with no partial fit method
        if not scalers.enable_partial:
            
            c_dataset, _ = PathLightDataset.collate(train_data)
            scalers.fit(c_dataset)
            
        # save scalers
        os.makedirs(scalers_folder, exist_ok=True)
        
        skdump(scalers, f'{scalers_folder}/scalers.bin', compress=True)

    # reload scalers    
    scalers = skload(f'{scalers_folder}/scalers.bin')
        
    transforms_list = [ScalerTransform(scalers)]
    
    if MIGNNConf.ENCODING is not None:
        print('[Scaling (with encoding)] start preparing encoded scaled data...')
        transforms_list.append(SignalEncoder(MIGNNConf.ENCODING, MIGNNConf.MASK))
    else:
        print('[Scaling] start preparing scaled data...')
    applied_transforms = GeoT.Compose(transforms_list)    

    # applied transformations over all intermediate path light dataset
    # avoid memory overhead
    if not os.path.exists(f'{dataset_path}/train'):
        
        output_scaled_temp_train = f'{output_temp_scaled}/train'
        os.makedirs(output_scaled_temp_train, exist_ok=True)
        
        output_scaled_temp_test = f'{output_temp_scaled}/test'
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
        train_dataset_path = f'{dataset_path}/train'
        test_dataset_path = f'{dataset_path}/test'
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
                 
        # print('[Cleaning] clear intermediates saved datasets...')
        # os.system(f'rm -r {output_temp}') 
        # os.system(f'rm -r {output_temp_scaled}') 
    else:
        print(f'[Information] {dataset_path} already generated')
 
if __name__ == "__main__":
    main()
