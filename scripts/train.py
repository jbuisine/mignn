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
    parser.add_argument('--data', type=str, help="output folder where (same when preparing data)", required=True)
    
    args = parser.parse_args()
    output_folder     = args.data

    # Some MIGNN params
    n_epochs          = MIGNNConf.EPOCHS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_folder = f'{output_folder}/model'
    stats_folder = f'{output_folder}/stats'
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)
    
    dataset_path = os.path.join(output_folder, 'train', 'datasets')
    print(f'[Loading] dataset from: `{dataset_path}`')
    
    train_folder = f'{dataset_path}_train'
    test_folder = f'{dataset_path}_test'
    
    # avoid metadata file
    dataset_train_paths = [ os.path.join(train_folder, p) for p in os.listdir(f'{dataset_path}_train') \
                    if 'metadata' not in p ]
    dataset_test_paths = [ os.path.join(test_folder, p) for p in os.listdir(f'{dataset_path}_test') \
                    if 'metadata' not in p ]
    
    train_info = json.load(open(f'{dataset_path}_train/metadata', 'r', encoding='utf-8'))
    test_info = json.load(open(f'{dataset_path}_test/metadata', 'r', encoding='utf-8'))
    
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
            
            train_dataset = PathLightDataset(root=f'{dataset_path}')
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

    def test(datasets, n_batchs):
        model.eval()

        error = 0
        r2_error = 0
        
        # test using multiple intermediate dataset and loader
        # Warn: need to respect number of batch (config batch size)
        for dataset_path in datasets:    
            
            dataset = PathLightDataset(root=f'{dataset_path}')
            loader = DataLoader(dataset, batch_size=MIGNNConf.BATCH_SIZE, shuffle=True)
        
            for data in loader:  # Iterate in batches over the training/test dataset.

                data = data.to(device)

                out = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)
                loss = criterion(out.flatten(), data.y.flatten())
                error += loss.item()
                r2_error += r2_score(out.flatten(), data.y.flatten()).item()
                
        return error / n_batchs, r2_error / n_batchs  # Derive ratio of correct predictions.

    stat_file = open(f'{stats_folder}/scores.csv', 'w', encoding='utf-8')
    stat_file.write('train_loss;train_r2;test_loss;test_r2\n')

    # save best only
    current_best_r2 = torch.tensor(float('-inf'), dtype=torch.float)
    for epoch in range(1, n_epochs + 1):
        train(epoch, dataset_train_paths, train_n_batchs)
        train_loss, train_r2 = test(dataset_train_paths, train_n_batchs)
        test_loss, test_r2 = test(dataset_test_paths, test_n_batchs)
        
        # save best only
        if test_r2 > current_best_r2:
            current_best_r2 = test_r2

            torch.save(model.state_dict(), f'{model_folder}/model.pt')
            torch.save(optimizer.state_dict(), f'{model_folder}/optimizer.pt')
            
        # save model stat data
        stat_file.write(f'{train_loss};{train_r2};{test_loss};{test_r2}\n')

        print(f'[Epoch n°{epoch:03d}]: Train ({MIGNNConf.LOSS} loss: {train_loss:.5f}, R²: {train_r2:.5f}), '\
            f'Test ({MIGNNConf.LOSS} loss: {test_loss:.5f}, R²: {test_r2:.5f})', end='\n')

    stat_file.close()

    print(f'[Saving] best model has been saved into: `{model_folder}`')

if __name__ == "__main__":
    main()
