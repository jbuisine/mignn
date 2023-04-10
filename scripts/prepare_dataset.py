import os
import argparse
import pickle

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

from mignn.dataset import PathLightDataset

import torch
import torch_geometric.transforms as GeoT

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from joblib import dump as skdump

import tqdm
from multiprocessing.pool import ThreadPool

from utils import prepare_data
from utils import load_sensor_from, load_build_and_stack, scale_subset

from transforms import ScalerTransform, SignalEncoder

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
    scaled_dataset_path = f'{output_folder}/train/datasets_scaled'

    model_folder = f'{output_folder}/model'
    os.makedirs(model_folder, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print('[Data generation] start generating GNN data using Mistuba3')
        gnn_folders, ref_images, _ = prepare_data(scene_file,
                                    max_depth = MIGNNConf.MAX_DEPTH,
                                    data_spp = MIGNNConf.GNN_SPP,
                                    ref_spp = MIGNNConf.REF_SPP,
                                    sensors = sensors,
                                    output_folder = f'{output_folder}/train/generated')


        # multiple datasets to avoid memory overhead
        output_temp = f'{output_folder}/train/temp/'
        os.makedirs(output_temp, exist_ok=True)
        
        # same for scaled datasets
        output_temp_scaled = f'{output_folder}/train/temp_scaled/'
        os.makedirs(output_temp_scaled, exist_ok=True)

        print('\n[Building connections] creating connections using Mistuba3')
        # multiprocess build of connections
        pool_obj = ThreadPool()

        # load in parallel same scene file, imply error. Here we load multiple scenes
        params = list(zip(gnn_folders,
                    [ scene_file for _ in range(len(gnn_folders)) ],
                    [ output_temp for _ in range(len(gnn_folders)) ],
                    ref_images))

        build_containers = []
        for result in tqdm.tqdm(pool_obj.imap(load_build_and_stack, params), total=len(params)):
            build_containers.append(result)

        # save intermediate PathLightDataset
        # Then fusion PathLightDatasets into only one
        intermediate_datasets = []
        # ensure file orders
        intermediate_datasets_path = sorted(os.listdir(output_temp))
        
        for dataset_name in intermediate_datasets_path:
            c_dataset_path = os.path.join(output_temp, dataset_name)
            c_dataset = PathLightDataset(root=c_dataset_path)
            intermediate_datasets.append(c_dataset)
            
        concat_datasets = torch.utils.data.ConcatDataset(intermediate_datasets)
        concatenated_dataset = PathLightDataset(dataset_path, concat_datasets)
        print(f'[Intermediate save] save computed dataset into: {dataset_path}')

        # merged_graph_container = LightGraphManager.fusion(build_containers)
        # print('[merged]', merged_graph_container)
            
        
        # prepare splitted dataset
        split_index = int(len(concatenated_dataset) * split_percent)
        train_dataset = concatenated_dataset[:split_index]
        test_dataset = concatenated_dataset[split_index:]
        
        print(f'[Information] dataset will be composed of {len(concatenated_dataset)} graphs (percent split: {split_percent})')
        print(f'[Processing] fit scalers from {len(train_dataset)} graphs')
        # Later use of pre_transform function of datasets in order to save data
        # normalize data
        x_scaler = MinMaxScaler().fit(train_dataset.data.x)
        edge_scaler = MinMaxScaler().fit(train_dataset.data.edge_attr)
        y_scaler = MinMaxScaler().fit(train_dataset.data.y.reshape(-1, 3))

        skdump(x_scaler, f'{model_folder}/x_node_scaler.bin', compress=True)
        skdump(edge_scaler, f'{model_folder}/x_edge_scaler.bin', compress=True)
        skdump(y_scaler, f'{model_folder}/y_scaler.bin', compress=True)

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
        if not os.path.exists(f'{scaled_dataset_path}.train'):
            
            # Potential NEW VERSION (does not well parallelized)
            intermediate_datasets_path = [ os.path.join(output_temp, p) for p in sorted(os.listdir(output_temp)) ]
            n_intermediates = len(intermediate_datasets_path)
            intermediate_scaled_datasets_path = []
            
            # multiprocess scale of dataset
            pool_obj_scaled = ThreadPool()

            # load in parallel same scene file, imply error. Here we load multiple scenes
            scaled_params = list(zip(intermediate_datasets_path,
                        [ model_folder for _ in range(n_intermediates) ],
                        [ output_temp_scaled for _ in range(n_intermediates) ]
                    ))
            
            for result in tqdm.tqdm(pool_obj_scaled.imap(scale_subset, scaled_params), total=len(scaled_params)):
                intermediate_scaled_datasets_path.append(result)
                
            intermediate_scaled_datasets = []
            for scaled_dataset_path in intermediate_scaled_datasets_path:
                intermediate_scaled_datasets.append(PathLightDataset(root=scaled_dataset_path, 
                                                        pre_transform=applied_transforms))
                
            # scaled_concat_datasets = torch.utils.data.ConcatDataset(intermediate_scaled_datasets)
            # scaled_concatenated_dataset = PathLightDataset(scaled_dataset_path, 
            #                                         scaled_concat_datasets)
            
            # Current VERSION
            # intermediate_scaled_datasets_path = []
            # intermediate_datasets_path = sorted(os.listdir(output_temp))
            
            # n_subsets = len(intermediate_datasets_path)
            # step = (n_subsets // 100) + 1
            
            # for idx, dataset_name in enumerate(intermediate_datasets_path):
            #     c_dataset_path = os.path.join(output_temp, dataset_name)
            #     c_dataset = PathLightDataset(root=c_dataset_path)
                
            #     c_scaled_dataset_path = os.path.join(output_temp_scaled, dataset_name)

            #     scaled_dataset = PathLightDataset(c_scaled_dataset_path, c_dataset, 
            #                         pre_transform=applied_transforms)
                
            #     if (idx % step == 0 or idx >= n_subsets - 1):
            #         print(f'[Scaling] -- progress: {(idx + 1) / n_subsets * 100.:.2f}%', \
            #             end='\r' if idx + 1 < n_subsets else '\n')
                
                # now transform dataset using scaler and encoding
                # intermediate_scaled_datasets_path.append(c_scaled_dataset_path)
                # del c_dataset # remove intermediate variable
                # del scaled_dataset
                
            # TODO: create TRAIN and TEST subsets
            # reload each scaled datasets
            # intermediate_scaled_datasets = []
            # for scaled_dataset_path in intermediate_scaled_datasets_path:
            #     intermediate_scaled_datasets.append(PathLightDataset(root=scaled_dataset_path, pre_transform=applied_transforms))

            # why need to remove before creating other dataset?
            os.system(f'rm -r {output_temp}') 
            os.system(f'rm -r {output_temp_scaled}') 
            
            scaled_concat_datasets = torch.utils.data.ConcatDataset(intermediate_scaled_datasets)
            scaled_concatenated_dataset = PathLightDataset(scaled_dataset_path, scaled_concat_datasets)
            
            train_dataset = scaled_concatenated_dataset[:split_index]
            test_dataset = scaled_concatenated_dataset[split_index:]
        
            # save scaled dataset
            print(f'[Saving] train and test datasets will be saved into: {dataset_path}')
            PathLightDataset(f'{dataset_path}.train', 
                            train_dataset)
            PathLightDataset(f'{dataset_path}.test', 
                            test_dataset)
        
            print('[Cleaning] clear intermediates saved datasets...')

            os.system(f'rm -r {scaled_dataset_path}') # remove also previous computed dataset
    else:
        print(f'[Information] {dataset_path} already generated')
 
if __name__ == "__main__":
    main()
