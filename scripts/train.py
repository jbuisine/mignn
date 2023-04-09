import os
import argparse

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

from mignn.dataset import PathLightDataset

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as GeoT

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from joblib import dump as skdump
from joblib import load as skload
from torchmetrics import R2Score

import tqdm
from multiprocessing.pool import ThreadPool

from models.gcn_model import GNNL

from utils import prepare_data
from utils import load_sensor_from, load_build_and_stack

from transforms import ScalerTransform, SignalEncoder

import config as MIGNNConf

def main():

    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--output', type=str, help="output folder", required=True)
    parser.add_argument('--name', type=str, help="output model name", required=True)
    parser.add_argument('--sensors', type=str, help="file with all viewpoints on scene", required=True)
    
    args = parser.parse_args()

    scene_file        = args.scene
    output_folder     = args.output
    model_name        = args.name
    sensors_folder    = args.sensors

    # Some MIGNN params 
    split_percent     = MIGNNConf.TRAINING_SPLIT
    w_size, h_size    = MIGNNConf.VIEWPOINT_SIZE
    sensors_n_samples = MIGNNConf.VIEWPOINT_SAMPLES
    n_epochs          = MIGNNConf.EPOCHS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    dataset_path = f'{output_folder}/train/datasets/{model_name}'
    # scaled_dataset_path = f'{output_folder}/train/datasets/{model_name}_scaled'

    model_folder = f'{output_folder}/models/{model_name}'
    stats_folder = f'{output_folder}/stats/{model_name}'
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)
    
    # empty dataset by default
    train_dataset = None
    test_dataset = None
    
    if not os.path.exists(dataset_path):
        gnn_folders, ref_images, _ = prepare_data(scene_file,
                                    max_depth = MIGNNConf.MAX_DEPTH,
                                    data_spp = MIGNNConf.GNN_SPP,
                                    ref_spp = MIGNNConf.REF_SPP,
                                    sensors = sensors,
                                    output_folder = f'{output_folder}/train/generated/{model_name}')


        # multiple datasets to avoid memory overhead
        output_temp = f'{output_folder}/train/temp/'
        os.makedirs(output_temp, exist_ok=True)
        
        # same for scaled datasets
        # output_temp_scaled = f'{output_folder}/train/temp_scaled/'
        # os.makedirs(output_temp_scaled, exist_ok=True)

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
        print(split_index)
        train_dataset = concatenated_dataset[:split_index]
        test_dataset = concatenated_dataset[split_index:]
        
        print(f'Dataset with {len(concatenated_dataset)} graphs (percent split: {split_percent})')
        
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
            print('[Encoded required] scaled data will be encoded')
            transforms_list.append(SignalEncoder(MIGNNConf.ENCODING))

        applied_transforms = GeoT.Compose(transforms_list)    

        # applied transformations over all intermediate path light dataset
        # avoid memory overhead
        # if not os.path.exists(f'{scaled_dataset_path}.train'):
            
        #     intermediate_scaled_datasets = []
        #     intermediate_datasets_path = sorted(os.listdir(output_temp))
            
        #     n_subsets = len(intermediate_datasets_path)
        #     step = (n_subsets // 100) + 1
            
        #     for idx, dataset_name in enumerate(intermediate_datasets_path):
        #         c_dataset_path = os.path.join(output_temp, dataset_name)
        #         c_dataset = PathLightDataset(root=c_dataset_path)
                
        #         c_scaled_dataset_path = os.path.join(output_temp_scaled, dataset_name)
        #         scaled_dataset = PathLightDataset(c_scaled_dataset_path, c_dataset, pre_transform=applied_transforms)
                
        #         if (idx % step == 0 or idx >= n_subsets - 1):
        #             print(f'[Scaling] -- progress: {(idx + 1) / n_subsets * 100.:.2f}%', \
        #                 end='\r' if idx + 1 < n_subsets else '\n')
                
        #         # now transform dataset using scaler and encoding
        #         intermediate_scaled_datasets.append(scaled_dataset)

        #     scaled_concat_datasets = torch.utils.data.ConcatDataset(intermediate_scaled_datasets)
        #     scaled_concatenated_dataset = PathLightDataset(scaled_dataset_path, scaled_concat_datasets)
        
            # save scaled dataset
        print(f'Save train and test dataset into: {dataset_path}')
        PathLightDataset(f'{dataset_path}.train', 
                        train_dataset, transform=applied_transforms)
        PathLightDataset(f'{dataset_path}.test', 
                        test_dataset, transform=applied_transforms)
    
        print('[cleaning] clear intermediates saved datasets...')
        os.system(f'rm -r {output_temp}')
            # os.system(f'rm -r {output_temp_scaled}')
            # os.system(f'rm -r {scaled_dataset_path}') # remove also previous computed dataset

    # need to reload scalers and transformers?
    x_scaler = skload(f'{model_folder}/x_node_scaler.bin')
    edge_scaler = skload(f'{model_folder}/x_edge_scaler.bin')
    y_scaler = skload(f'{model_folder}/y_scaler.bin')

    scalers = {
        'x_node': x_scaler,
        'x_edge': edge_scaler,
        'y': y_scaler
    }

    transforms_list = [ScalerTransform(scalers)]
    
    if MIGNNConf.ENCODING is not None:
        transforms_list.append(SignalEncoder(MIGNNConf.ENCODING))

    applied_transforms = GeoT.Compose(transforms_list)    

    print(f'Load scaled dataset from: `{dataset_path}.train` and `{dataset_path}.test`')
    
    # TODO: use of GPU based dataset?
    if train_dataset is None or test_dataset is None:
        train_dataset = PathLightDataset(root=f'{dataset_path}.train', transform=applied_transforms)
        test_dataset = PathLightDataset(root=f'{dataset_path}.test', transform=applied_transforms)
    print(f'Example of scaled element from train dataset: {train_dataset[0]}')

    train_loader = DataLoader(train_dataset, batch_size=MIGNNConf.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=MIGNNConf.BATCH_SIZE, shuffle=True)

    print('Prepare model: ')
    model = GNNL(hidden_channels=MIGNNConf.HIDDEN_CHANNELS, n_features=train_dataset.num_node_features).to(device)
    # model.to(device)
    print(model)
    print(f'Number of params: {sum(p.numel() for p in model.parameters())}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    r2_score = R2Score().to(device)

    def train(epoch_id):
        model.train()

        error = 0
        r2_error = 0
        for b_i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.

            data = data.to(device)

            out = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)  # Perform a single forward pass.
            loss = criterion(out.flatten(), data.y.flatten())  # Compute the loss.
            error += loss.item()
            loss.backward()  # Derive gradients.
            r2_error += r2_score(out.flatten(), data.y.flatten()).item()
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

            print(f'[Epoch n°{epoch_id:03d}] -- progress: {(b_i + 1) / len(train_loader) * 100.:.2f}%' \
                f' (Loss: {error / (b_i + 1):.5f}, R²: {r2_error / (b_i + 1):.5f})', end='\r')

    def test(loader):
        model.eval()

        error = 0
        r2_error = 0
        for data in loader:  # Iterate in batches over the training/test dataset.

            data = data.to(device)

            out = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)
            loss = criterion(out.flatten(), data.y.flatten())
            error += loss.item()
            r2_error += r2_score(out.flatten(), data.y.flatten()).item()
        return error / len(loader), r2_error / len(loader)  # Derive ratio of correct predictions.

    stat_file = open(f'{stats_folder}/{model_name}.csv', 'w', encoding='utf-8')
    stat_file.write('train_loss;train_r2;test_loss;test_r2\n')

    # save best only
    current_best_r2 = torch.tensor(float('-inf'), dtype=torch.float)
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        train_loss, train_r2 = test(train_loader)
        test_loss, test_r2 = test(test_loader)
        
        # save best only
        if test_r2 > current_best_r2:
            current_best_r2 = test_r2

            torch.save(model.state_dict(), f'{model_folder}/model.pt')
            torch.save(optimizer.state_dict(), f'{model_folder}/optimizer.pt')
            
        # save model stat data
        stat_file.write(f'{train_loss};{train_r2};{test_loss};{test_r2}\n')

        print(f'[Epoch n°{epoch:03d}]: Train (Loss: {train_loss:.5f}, R²: {train_r2:.5f}), '\
            f'Test (Loss: {test_loss:.5f}, R²: {test_r2:.5f})', end='\n')

    stat_file.close()

    print(f'Model has been saved into: `{model_folder}`')

if __name__ == "__main__":
    main()
