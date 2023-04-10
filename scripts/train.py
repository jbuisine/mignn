import os
import argparse

from mignn.dataset import PathLightDataset

import torch
from torch_geometric.loader import DataLoader

from torchmetrics import R2Score

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
    print(f'Load scaled dataset from: `{dataset_path}.train` and `{dataset_path}.test`')
    train_dataset = PathLightDataset(root=f'{dataset_path}.train')
    test_dataset = PathLightDataset(root=f'{dataset_path}.test')
        
    print(f'Train dataset: {len(train_dataset)} graphs')
    print(f'Test dataset: {len(test_dataset)} graphs')
    print(f'Example of graph from train dataset: {train_dataset[0]}')
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

    stat_file = open(f'{stats_folder}/scores.csv', 'w', encoding='utf-8')
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
