import os
import argparse
import json
import random

import torch
from torch_geometric.loader import DataLoader
from torchmetrics import R2Score

from mignn.dataset import PathLightDataset

from utils import init_loss
from models.gcn_model import GNNL
import config as MIGNNConf


def main():

    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--dataset', type=str, help="output folder where (same when preparing data)", required=True)
    parser.add_argument('--output', type=str, help="output model folder", required=True)
    
    args = parser.parse_args()
    dataset_path     = args.dataset
    output_folder     = args.output

    # Some MIGNN params
    n_epochs          = MIGNNConf.EPOCHS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_folder = f'{output_folder}/model'
    stats_folder = f'{output_folder}/stats'
    os.makedirs(model_folder, exist_ok=True)
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
    test_n_batchs = int(test_info['n_batchs'])
    
    print(f'[Information] train dataset composed of: {train_info["n_samples"]} graphs')
    print(f'[Information] test dataset composed of: {test_info["n_samples"]} graphs')
    
    print('[Information] model architecture:')
    n_node_features = int(train_info['n_node_features'])
    model = GNNL(hidden_channels=MIGNNConf.HIDDEN_CHANNELS, n_features=n_node_features).to(device)
    
    print(model)
    print(f'[Information] model with number of params: {sum(p.numel() for p in model.parameters())}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = init_loss(MIGNNConf.LOSS)
    r2_score = R2Score().to(device)

    def train(epoch_id, datasets, n_batchs):
        model.train()

        error = 0
        r2_error = 0
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

                out = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)  # Perform a single forward pass.
                loss = criterion(out.flatten(), data.y.flatten())  # Compute the loss.
                error += loss.item()
                loss.backward()  # Derive gradients.
                r2_error += r2_score(out.flatten(), data.y.flatten()).item()
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.

                print(f'[Epoch n°{epoch_id:03d}] -- progress: {(b_i + 1) / n_batchs * 100.:.2f}%' \
                    f' ({MIGNNConf.LOSS} loss: {error / (b_i + 1):.5f}, R²: {r2_error / (b_i + 1):.5f})', end='\r')
                b_i += 1
                
        return error / n_batchs, r2_error / n_batchs

    def test(datasets, n_batchs):
        model.eval()

        error = 0
        r2_error = 0
        
        # test using multiple intermediate dataset and loader
        # Warn: need to respect number of batch (config batch size)
        for dataset_path in datasets:    
            
            dataset = PathLightDataset(root=dataset_path)
            loader = DataLoader(dataset, batch_size=MIGNNConf.BATCH_SIZE, shuffle=True)
        
            for data in loader:  # Iterate in batches over the training/test dataset.

                data = data.to(device)

                out = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)
                loss = criterion(out.flatten(), data.y.flatten())
                error += loss.item()
                r2_error += r2_score(out.flatten(), data.y.flatten()).item()
                
        return error / n_batchs, r2_error / n_batchs

    stat_file = open(f'{stats_folder}/scores.csv', 'w', encoding='utf-8')
    stat_file.write('train_loss;train_r2;test_loss;test_r2\n')

    # reload model if necessary
    start_epoch = 1
    current_best_epoch = 1
    
    # save best only
    current_best_r2 = torch.tensor(float('-inf'), dtype=torch.float)
    
    model_params_filename = f'{model_folder}/model.pt'
    optimizer_params_filename = f'{model_folder}/optimizer.pt'
    
    # reload model data
    if os.path.exists(model_params_filename):
        
        model.load_state_dict(torch.load(model_params_filename))
        optimizer.load_state_dict(torch.load(optimizer_params_filename))
        
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
        train_loss, train_r2 = train(epoch, dataset_train_paths, train_n_batchs)
        # train_loss, train_r2 = test(dataset_train_paths, train_n_batchs)
        test_loss, test_r2 = test(dataset_test_paths, test_n_batchs)
        
        # save best only
        if test_r2 > current_best_r2:
            current_best_r2 = test_r2

            torch.save(model.state_dict(), model_params_filename)
            torch.save(optimizer.state_dict(), optimizer_params_filename)
            
        # save number of epochs done
        metadata = { 'epoch': epoch, 'best_r2': test_r2, 'best_epoch': current_best_epoch }
        with open(f'{model_folder}/metadata', 'w', encoding='utf-8') as outfile:
            json.dump(metadata, outfile)
            
        # save model stat data
        stat_file.write(f'{train_loss};{train_r2};{test_loss};{test_r2}\n')

        print(f'[Epoch n°{epoch:03d}]: Train ({MIGNNConf.LOSS} loss: {train_loss:.5f}, R²: {train_r2:.5f}), '\
            f'Test ({MIGNNConf.LOSS} loss: {test_loss:.5f}, R²: {test_r2:.5f})', end='\n')

    stat_file.close()

    print(f'[Saving] best model has been saved into: `{model_folder}`')

if __name__ == "__main__":
    main()
