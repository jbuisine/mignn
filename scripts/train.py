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
from models.nerf import BasicNeRf
import config as MIGNNConf


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
    gnn_model = GNNL(hidden_channels=MIGNNConf.HIDDEN_CHANNELS, n_features=n_node_features).to(device)
    
    # compute number of features for NeRF
    enc_mask, enc_size = MIGNNConf.ENCODING_MASK, MIGNNConf.ENCODING_SIZE
    nerf_features = sum(enc_mask['origin']) * enc_size * 2 + sum(enc_mask['origin']) \
        + sum(enc_mask['direction']) * enc_size * 2 + sum(enc_mask['direction'])
        
    print("n_features [NeRF]: ", nerf_features)
    nerf_model = BasicNeRf(nerf_features, MIGNNConf.NERF_LAYER_SIZE, MIGNNConf.NERF_HIDDEN_LAYERS).to(device)
    
    print('\nNeRF direct randiance Model:')
    print(nerf_model)
    
    print('GNN indirect radiance Model:')
    print(gnn_model)
    print(f'[Information] GNN model with number of params: {sum(p.numel() for p in gnn_model.parameters())}')
    print(f'[Information] NeRF model with number of params: {sum(p.numel() for p in nerf_model.parameters())}')

    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    nerf_optimizer = torch.optim.Adam(nerf_model.parameters(), lr=0.001)
    criterion = init_loss(MIGNNConf.LOSS).to(device)
    r2_score = R2Score().to(device)

    def train(epoch_id, datasets, n_batchs):
        gnn_model.train()
        nerf_model.train()

        gnn_error = 0
        nerf_error = 0
        global_error = 0
        global_r2_error = 0
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
                nerf_optimizer.zero_grad()
                
                nerf_input = torch.cat([data.origin, data.direction], dim=1).to(device)
                
                o_direct_radiance = nerf_model(nerf_input)
                # o_direct_radiance = o_direct_radiance[:, :-1] * o_direct_radiance[:, -1]
                loss = criterion(o_direct_radiance.flatten(), data.y_direct.flatten())  # Compute the loss.
                nerf_error += loss.item()
                loss.backward()
                nerf_optimizer.step()  
                
                
                gnn_optimizer.zero_grad() 
                
                # TODO: predict also for each graph the camera and direction position
                o_indirect_radiance = gnn_model(data.x, data.edge_attr, data.edge_index, batch=data.batch) 
                loss = criterion(o_indirect_radiance.flatten(), data.y_indirect.flatten())
                gnn_error += loss.item()
                loss.backward()
                gnn_optimizer.step()  
                
                expected_radiance = data.y_direct.flatten() + data.y_indirect.flatten()
                o_radiance = o_direct_radiance.flatten() + o_indirect_radiance.flatten()
                
                global_error += criterion(o_radiance, expected_radiance).item()
                global_r2_error += r2_score(o_radiance, expected_radiance).item()

                print(f'[Epoch n°{epoch_id:03d}] -- progress: {(b_i + 1) / n_batchs * 100.:.2f}%' \
                    f' ([{MIGNNConf.LOSS}] nerf: {nerf_error / (b_i + 1):.3f}, gnn: {gnn_error / (b_i + 1):.3f}, '\
                    f'global: {global_error / (b_i + 1):.5f}, ' \
                    f'R²: {global_r2_error / (b_i + 1):.5f})', end='\r')
                b_i += 1
                
        return nerf_error / n_batchs, gnn_error / n_batchs, global_error / n_batchs, global_r2_error / n_batchs

    def test(datasets, n_batchs):
        gnn_model.eval()
        nerf_model.eval()

        gnn_error = 0
        nerf_error = 0
        global_error = 0
        global_r2_error = 0
        
        # test using multiple intermediate dataset and loader
        # Warn: need to respect number of batch (config batch size)
        for dataset_path in datasets:    
            
            dataset = PathLightDataset(root=dataset_path)
            loader = DataLoader(dataset, batch_size=MIGNNConf.BATCH_SIZE, shuffle=True)
        
            for data in loader:  # Iterate in batches over the training/test dataset.

                data = data.to(device)

                nerf_input = torch.cat([data.origin, data.direction], dim=1).to(device)
                o_direct_radiance = nerf_model(nerf_input)
                # o_direct_radiance = o_direct_radiance[:, :-1] * o_direct_radiance[:, -1]
                loss = criterion(o_direct_radiance.flatten(), data.y_direct.flatten())  # Compute the loss.
                nerf_error += loss.item()
                
                # TODO: predict also for each graph the camera and direction position
                o_indirect_radiance = gnn_model(data.x, data.edge_attr, data.edge_index, batch=data.batch) 
                loss = criterion(o_indirect_radiance.flatten(), data.y_indirect.flatten())
                gnn_error += loss.item()
                
                expected_radiance = data.y_direct.flatten() + data.y_indirect.flatten()
                o_radiance = o_direct_radiance.flatten() + o_indirect_radiance.flatten()
                
                global_error += criterion(o_radiance, expected_radiance).item()
                global_r2_error += r2_score(o_radiance, expected_radiance).item()
                
        return nerf_error / n_batchs, gnn_error / n_batchs, global_error / n_batchs, global_r2_error / n_batchs

    stat_file = open(f'{stats_folder}/scores.csv', 'w', encoding='utf-8')
    stat_file.write('train_loss;train_r2;test_loss;test_r2\n')

    # reload model if necessary
    start_epoch = 1
    current_best_epoch = 1
    
    # save best only
    current_best_r2 = torch.tensor(float('-inf'), dtype=torch.float)
    
    gnn_model_params_filename = f'{model_folder}/model_gnn.pt'
    nerf_model_params_filename = f'{model_folder}/model_nerf.pt'
    gnn_optimizer_params_filename = f'{model_folder}/optimizer_gnn.pt'
    nerf_optimizer_params_filename = f'{model_folder}/optimizer_nerf.pt'
    
    # reload model data
    if os.path.exists(gnn_model_params_filename):
        
        gnn_model.load_state_dict(torch.load(gnn_model_params_filename))
        nerf_model.load_state_dict(torch.load(nerf_model_params_filename))
        gnn_optimizer.load_state_dict(torch.load(gnn_optimizer_params_filename))
        nerf_optimizer.load_state_dict(torch.load(nerf_optimizer_params_filename))
        
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
        train_nerf_loss, train_gnn_loss, train_loss, train_r2 = train(epoch, dataset_train_paths, train_n_batchs)
        # train_loss, train_r2 = test(dataset_train_paths, train_n_batchs)
        test_nerf_loss, test_gnn_loss, test_loss, test_r2 = test(dataset_test_paths, test_n_batchs)
        
        # save best only
        if test_r2 > current_best_r2:
            current_best_r2 = test_r2
            current_best_epoch = epoch

            torch.save(gnn_model.state_dict(), gnn_model_params_filename)
            torch.save(nerf_model.state_dict(), nerf_model_params_filename)
            torch.save(gnn_optimizer.state_dict(), gnn_optimizer_params_filename)
            torch.save(nerf_optimizer.state_dict(), nerf_optimizer_params_filename)
            
        # save number of epochs done
        metadata = { 'epoch': epoch, 'best_r2': test_r2, 'best_epoch': current_best_epoch }
        with open(f'{model_folder}/metadata', 'w', encoding='utf-8') as outfile:
            json.dump(metadata, outfile)
            
        # save model stat data
        stat_file.write(f'{train_loss};{train_r2};{test_loss};{test_r2}\n')

        print(f'[Epoch n°{epoch:03d}]: Train ([{MIGNNConf.LOSS}] nerf: {train_nerf_loss:.5f}, gnn: {train_gnn_loss:.5f}, '
            f'global: {train_loss:.5f}, R²: {train_r2:.5f}), '\
            f'Test ([{MIGNNConf.LOSS}] nerf: {test_nerf_loss:.3f}, gnn: {test_gnn_loss:.3f} '\
            f'global: {test_loss:.5f}, R²: {test_r2:.5f})', end='\n')

    stat_file.close()

    print(f'[Saving] best model has been saved into: `{model_folder}`')

if __name__ == "__main__":
    main()
