import os
import argparse
import json
import random

import torch
from torch_geometric.loader import DataLoader

from mignn.dataset import PathLightDataset

import config as MIGNNConf
from models.manager import ManagerFactory
from models.param import ModelParam

def main():

    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--dataset', type=str, help="output folder where (same when preparing data)", required=True)
    parser.add_argument('--output', type=str, help="output model folder", required=True)
    
    args = parser.parse_args()
    dataset_path     = args.dataset
    output_folder    = args.output

    # Some MIGNN params
    n_epochs         = MIGNNConf.EPOCHS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_folder = f'{output_folder}/model'
    stats_folder = f'{output_folder}/stats'
    os.makedirs(stats_folder, exist_ok=True)
    
    print(f'[Loading] dataset from: `{dataset_path}`')
    
    train_folder = f'{dataset_path}/data/train'
    test_folder = f'{dataset_path}/data/test'
    
    # avoid metadata file
    dataset_train_paths = [ os.path.join(train_folder, p) for p in os.listdir(train_folder) \
                    if 'metadata' not in p ]
    dataset_test_paths = [ os.path.join(test_folder, p) for p in os.listdir(test_folder) \
                    if 'metadata' not in p ]
    
    train_info = json.load(open(f'{dataset_path}/data/train/metadata', 'r', encoding='utf-8'))
    test_info = json.load(open(f'{dataset_path}/data/test/metadata', 'r', encoding='utf-8'))
    
    train_n_batchs = int(train_info['n_batchs'])
    # test_n_batchs = int(test_info['n_batchs'])
    
    print(f'[Information] train dataset composed of: {train_info["n_samples"]} graphs')
    print(f'[Information] test dataset composed of: {test_info["n_samples"]} graphs')
    
    print('[Information] model architecture:')
    
    # camera features size
    enc_mask, enc_size = MIGNNConf.ENCODING_MASK, MIGNNConf.ENCODING_SIZE
    n_camera_features = sum(enc_mask['origin']) * enc_size * 2 + sum(enc_mask['origin']) \
        + sum(enc_mask['direction']) * enc_size * 2 + sum(enc_mask['direction'])
        
    n_node_features = int(train_info['n_node_features'])
    
    model_manager = ManagerFactory.create(n_node_features, n_camera_features, MIGNNConf)
    
    # END INSTANTIATE THE MODEL MANAGER
    for kind, model in model_manager.models.items():
        print(f'[Information] {kind} model with number of params: {sum(p.numel() for p in model.parameters())}')

    def train(epoch_id, datasets, n_batchs):
        model_manager.train()

        b_i = 0
        
        # random datasets to avoid same subset order
        random.shuffle(datasets)
        
        # train using multiple intermediate dataset and loader
        # Warn: need to respect number of batch (config batch size)
        for dataset_path in datasets:    
            
            train_dataset = PathLightDataset(root=dataset_path)
            train_loader = DataLoader(train_dataset, batch_size=MIGNNConf.BATCH_SIZE, shuffle=True)
            
            for data in train_loader:  # Iterate in batches over the training dataset.

                data = data.to(device)
                
                # TODO: DO STEP using manager
                model_manager.step(data)
                
                print(f'[Epoch n°{epoch_id:03d}] -- progress: {(b_i + 1) / n_batchs * 100.:.2f}% -- {model_manager.information("train")}', end='\r')
                
                b_i += 1
                

    def test(datasets):
        model_manager.eval()
        
        # test using multiple intermediate dataset and loader
        # Warn: need to respect number of batch (config batch size)
        for dataset_path in datasets:    
            
            dataset = PathLightDataset(root=dataset_path)
            loader = DataLoader(dataset, batch_size=MIGNNConf.BATCH_SIZE, shuffle=True)
        
            for data in loader:  # Iterate in batches over the training/test dataset.

                data = data.to(device)
                model_manager.test(data)
                

    stat_file = open(f'{stats_folder}/scores.csv', 'w', encoding='utf-8')
    header = model_manager.metrics_header()
    stat_file.write(f'{";".join(header)}\n')

    # reload model if necessary
    start_epoch = 1
    current_best_epoch = 1
    
    # save best only
    current_best_r2 = torch.tensor(float('-inf'), dtype=torch.float)
    
    # reload model data
    if os.path.exists(model_folder):
        
        # load model using model manager
        model_manager.load(model_folder)
        
        train_metadata = json.load(open(f'{model_folder}/metadata', 'r', encoding='utf-8'))
        start_epoch = int(train_metadata['epoch'])
        start_epoch = 0 if start_epoch < 0 else start_epoch # ensure non negative epoch
        
        current_best_r2 = float(train_metadata['best_r2'])
        current_best_epoch = int(train_metadata['best_epoch'])
        
        print(f'[Information] load previous best saved model at epoch {start_epoch}')
        print(f'[Information] model had R²: {current_best_r2} on test dataset (@epoch n°{current_best_epoch})')
    
    if start_epoch == n_epochs:
        print('[Information] no need to futhermore train model')
        return
    
    for epoch in range(start_epoch, n_epochs + 1):
        
        train(epoch, dataset_train_paths, train_n_batchs)
        test(dataset_test_paths)
        
        test_r2 = model_manager.score('test')
        
        # save best only
        if test_r2 > current_best_r2:
            current_best_r2 = test_r2
            current_best_epoch = epoch
                            
            # save using the model manager
            model_manager.save(model_folder)    
            
        # save number of epochs done
        metadata = { 'epoch': epoch, 'best_r2': test_r2, 'best_epoch': current_best_epoch }
        with open(f'{model_folder}/metadata', 'w', encoding='utf-8') as outfile:
            json.dump(metadata, outfile)
            
        # save model stat data
        metrics = model_manager.metrics_values()
        stat_file.write(f'{";".join(list(map(str, metrics)))}\n')

        print(f'[Epoch n°{epoch:03d}]: train: {model_manager.information("train")}, test: {model_manager.information("test")}')
        model_manager.clear_metrics()
        

    stat_file.close()

    print(f'[Saving] best model has been saved into: `{model_folder}`')

if __name__ == "__main__":
    main()
