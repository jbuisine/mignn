import os
import argparse
import dill

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

from mignn.container import SimpleLightGraphContainer
from mignn.manager import LightGraphManager
from mignn.dataset import PathLightDataset
from mignn.processing.encoder import signal_encoder

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump as skdump
from torchmetrics import R2Score

import tqdm
from multiprocessing.pool import ThreadPool

from models.gcn_model import GNNL

from utils import prepare_data, scale_data
from utils import load_sensor_from, load_build_and_stack


def main():

    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--output', type=str, help="output folder", required=True)
    parser.add_argument('--name', type=str, help="output model name", required=True)
    parser.add_argument('--epochs', type=int, help="expected number of epochs", required=False, default=10)
    parser.add_argument('--encoder', type=int, help="encoding data or not", required=False, default=False)
    parser.add_argument('--encoder_size', type=int, help="encoding size per feature", required=False, default=6)
    parser.add_argument('--sensors', type=str, help="file with all viewpoints on scene", required=True)
    parser.add_argument('--nsamples', type=int, help="Number of GNN file sample per sensor", default=1)
    parser.add_argument('--split', type=float, help="split percent \in [0, 1]", required=False, default=0.8)
    parser.add_argument('--img_size', type=str, help="expected computed image size: 128,128", required=False, default="128,128")

    args = parser.parse_args()

    scene_file        = args.scene
    output_folder     = args.output
    model_name        = args.name
    n_epochs          = args.epochs
    encoder_enabled   = args.encoder
    split_percent     = args.split
    sensors_folder    = args.sensors
    sensors_n_samples = args.nsamples
    w_size, h_size    = list(map(int, args.img_size.split(',')))
    encoder_size      = args.encoder_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use of: https://github.com/prise-3d/vpbrt
    # read from camera LookAt folder
    sensors = []
    for file in sorted(os.listdir(sensors_folder)):
        file_path = os.path.join(sensors_folder, file)

        sensor = load_sensor_from((w_size, h_size), file_path)

        # Use a number of times the same sensors in order to increase knowledge
        # Multiple GNN files will be generated
        for i in range(sensors_n_samples):
            sensors.append(sensor)

    os.makedirs(output_folder, exist_ok=True)
    dataset_path = f'{output_folder}/train/datasets/{model_name}'

    if not os.path.exists(dataset_path):
        gnn_files, ref_images, _ = prepare_data(scene_file,
                                    max_depth = 5,
                                    data_spp = 10,
                                    ref_spp = 10000,
                                    sensors = sensors,
                                    output_folder = f'{output_folder}/train/generated/{model_name}')



        output_temp = f'{output_folder}/train/temp/'
        os.makedirs(output_temp, exist_ok=True)

        # multiprocess build of connections
        pool_obj = ThreadPool()

        # load in parallel same scene file, imply error. Here we load multiple scenes
        params = list(zip(gnn_files,
                    [ scene_file for _ in range(len(gnn_files)) ],
                    [ output_temp for _ in range(len(gnn_files)) ],
                    ref_images))

        build_containers = []
        for result in tqdm.tqdm(pool_obj.imap(load_build_and_stack, params), total=len(params)):
            build_containers.append(result)

        # fusion pixels grids (note here: each container correspond to a viewpoint)
        # avoid to vstack now (loss of individual viewpoints)
        merged_graph_container = LightGraphManager.fusion(build_containers)
        print('[merged]', merged_graph_container)

        print(f'[cleaning] clear intermediated saved containers into {output_temp}')
        os.system(f'rm -r {output_temp}')

        dataset = PathLightDataset.from_container(merged_graph_container, dataset_path)
        print(f'[Intermediate save] save computed dataset into: {dataset_path}')

    dataset = PathLightDataset(root=dataset_path)
    print(f'Dataset with {len(dataset)} graphs (percent split: {split_percent})')

    split_index = int(len(dataset) * split_percent)
    train_dataset = dataset[:split_index]
    test_dataset = dataset[split_index:]

    model_folder = f'{output_folder}/models/{model_name}'
    stats_folder = f'{output_folder}/stats/{model_name}'
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)

    # normalize data
    x_scaler = MinMaxScaler().fit(train_dataset.data.x)
    edge_scaler = MinMaxScaler().fit(train_dataset.data.edge_attr)
    # y_scaler = MinMaxScaler().fit(train_dataset.data.y.reshape((-1, 3)))

    skdump(x_scaler, f'{model_folder}/x_node_scaler.bin', compress=True)
    skdump(edge_scaler, f'{model_folder}/x_edge_scaler.bin', compress=True)
    # skdump(y_scaler, f'{model_folder}/y_scaler.bin', compress=True)

    if encoder_enabled:
        print('[Encoded required] scaled data will be encoded')

    scaled_dataset_path = f'{output_folder}/train/datasets/{model_name}_scaled'

    if not os.path.exists(f'{scaled_dataset_path}.train'):

        scalers = {
            'x_node': x_scaler,
            'x_edge': edge_scaler
        }

        scaled_data_list = scale_data(dataset, scalers, encoder_enabled, encoder_size)

        # save dataset
        print(f'Save scaled train and test dataset into: {scaled_dataset_path}')
        PathLightDataset(f'{scaled_dataset_path}.train', scaled_data_list[:split_index])
        PathLightDataset(f'{scaled_dataset_path}.test', scaled_data_list[split_index:])

    print(f'Load scaled dataset from: `{scaled_dataset_path}.train` and `{scaled_dataset_path}.test`')
    train_dataset = PathLightDataset(root=f'{scaled_dataset_path}.train')
    test_dataset = PathLightDataset(root=f'{scaled_dataset_path}.test')
    print(f'Example of scaled element from train dataset: {train_dataset[0]}')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    print('Prepare model: ')
    model = GNNL(hidden_channels=256, n_features=train_dataset.num_node_features).to(device)
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
            loss = criterion(out.flatten(), data.y)  # Compute the loss.
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
            loss = criterion(out.flatten(), data.y)
            error += loss.item()
            r2_error += r2_score(out.flatten(), data.y.flatten()).item()
        return error / len(loader), r2_error / len(loader)  # Derive ratio of correct predictions.

    stat_file = open(f'{stats_folder}/{model_name}.csv', 'w', encoding='utf-8')
    stat_file.write('train_loss;train_r2;test_loss;test_r2\n')

    for epoch in range(1, n_epochs + 1):
        train(epoch)
        train_loss, train_r2 = test(train_loader)
        test_loss, test_r2 = test(test_loader)

        # save model stat data
        stat_file.write(f'{train_loss};{train_r2};{test_loss};{test_r2}\n')

        print(f'[Epoch n°{epoch:03d}]: Train (Loss: {train_loss:.5f}, R²: {train_r2:.5f}), '\
            f'Test (Loss: {test_loss:.5f}, R²: {test_r2:.5f})', end='\n')

    stat_file.close()

    torch.save(model.state_dict(), f'{model_folder}/model.pt')
    torch.save(optimizer.state_dict(), f'{model_folder}/optimizer.pt')

    print(f'Model has been saved into: `{model_folder}`')

if __name__ == "__main__":
    main()
