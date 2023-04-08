import os
import argparse
import numpy as np

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


from config import REF_SPP, GNN_SPP, MAX_DEPTH

from mignn.dataset import PathLightDataset

import torch
from joblib import load as skload

from models.gcn_model import GNNL

from utils import scale_data, prepare_data
from utils import load_sensor_from, load_build_and_stack
import matplotlib.pyplot as plt

import tqdm
from multiprocessing.pool import ThreadPool
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM

# ignore Drjit warning
import warnings
warnings.filterwarnings('ignore')

def main():

    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--model', type=str, help="where to find saved model", required=True)
    parser.add_argument('--output', type=str, help="output data folder", required=True)
    parser.add_argument('--encoder', type=int, help="encoding data or not", required=False, default=False)
    parser.add_argument('--encoder_size', type=int, help="encoding size per feature", required=False, default=6)
    parser.add_argument('--sensors', type=str, help="specific sensors folder", required=True)
    parser.add_argument('--img_size', type=str, help="expected computed image size: 128,128", required=False, default="128,128")

    args = parser.parse_args()

    scene_file        = args.scene
    model_folder      = args.model
    output_folder     = args.output
    encoder_enabled   = args.encoder
    encoder_size      = args.encoder_size
    sensors_folder    = args.sensors
    w_size, h_size    = list(map(int, args.img_size.split(',')))

    # use of: https://github.com/prise-3d/vpbrt
    # read from camera LookAt folder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use of: https://github.com/prise-3d/vpbrt
    # read from camera LookAt folder
    sensors = []
    viewpoints = []
    for file in sorted(os.listdir(sensors_folder)):
        file_path = os.path.join(sensors_folder, file)
        viewpoint_name = file.split('.')[0]
        sensor = load_sensor_from((w_size, h_size), file_path)
        sensors.append(sensor)
        viewpoints.append(viewpoint_name)


    predictions = []
    references = []
    low_res_images = []

    gnn_files, ref_images, low_images = prepare_data(scene_file,
                            max_depth = MAX_DEPTH,
                            data_spp = GNN_SPP,
                            ref_spp = REF_SPP,
                            sensors = sensors,
                            output_folder = f'{output_folder}/generated')

    output_temp = f'{output_folder}/datasets/temp/'
    os.makedirs(output_temp, exist_ok=True)

    # multiprocess build of connections
    pool_obj = ThreadPool()

    # load in parallel same scene file, imply error. Here we load multiple scenes
    params = list(zip(gnn_files,
                [ scene_file for _ in range(len(gnn_files)) ],
                [ os.path.join(output_temp, v) for v in viewpoints ],
                ref_images))

    build_containers = []
    for result in tqdm.tqdm(pool_obj.imap(load_build_and_stack, params), total=len(params)):
        build_containers.append(result)

    datasets_path = []
    for v_i, viewpoint in enumerate(viewpoints):
        
        viewpoint_temp = os.path.join(output_temp, viewpoint)
        intermediate_datasets = []
        intermediate_datasets_path = os.listdir(viewpoint_temp)
        
        for dataset_name in intermediate_datasets_path:
            c_dataset_path = os.path.join(viewpoint_temp, dataset_name)
            c_dataset = PathLightDataset(root=c_dataset_path)
            intermediate_datasets.append(c_dataset)
            
        dataset_path = f'{output_folder}/datasets/{viewpoint}'        
        dataset = PathLightDataset.fusion(intermediate_datasets, dataset_path)
        print(f' -- [Intermediate save] save computed dataset into: {dataset_path}')
        datasets_path.append(dataset_path)

    print(f'[cleaning] clear intermediated saved containers into {output_temp}')
    os.system(f'rm -r {output_temp}')

    for v_i, sensor in enumerate(sensors):

        dataset_path = datasets_path[v_i]
        viewpoint_name = viewpoints[v_i]
        v_ref_image = np.asarray(cv2.imread(ref_images[v_i], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        v_low_image = np.asarray(cv2.imread(low_images[v_i], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))

        print(f'[Manage viewpoint n°{v_i}: {viewpoint_name}]')

        dataset = PathLightDataset(root=dataset_path)

        x_scaler = skload(f'{model_folder}/x_node_scaler.bin')
        edge_scaler = skload(f'{model_folder}/x_edge_scaler.bin')
        y_scaler = skload(f'{model_folder}/y_scaler.bin')

        scaled_dataset_path = f'{output_folder}/datasets/{viewpoint_name}_scaled'

        if encoder_enabled:
            print(' -- [Encoded required] scaled data will be encoded')

        if not os.path.exists(scaled_dataset_path):

            scalers = {
                'x_node': x_scaler,
                'x_edge': edge_scaler,
                'y': y_scaler
            }

            scaled_data_list = scale_data(dataset, scalers, encoder_enabled, encoder_size)

            # save dataset
            print(f' -- Save scaled dataset into: {scaled_dataset_path}')
            PathLightDataset(scaled_dataset_path, scaled_data_list)

        print(f' -- Load scaled dataset from: {scaled_dataset_path}')
        dataset = PathLightDataset(root=scaled_dataset_path)

        model = GNNL(hidden_channels=256, n_features=dataset.num_node_features)
        print(' -- Model has been loaded')

        model.load_state_dict(torch.load(f'{model_folder}/model.pt'))
        model.eval()

        pixels = []

        n_predictions = len(dataset)
        for b_i in range(n_predictions):

            data = dataset[b_i]
            prediction = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)
            prediction = y_scaler.inverse_transform(prediction.detach().numpy())
            pixels.append(prediction)

            print(f' -- Prediction progress: {(b_i + 1) / len(dataset) * 100.:.2f}%', end='\r')

        image = np.array(pixels).reshape((h_size, w_size, 3))

        os.makedirs(f'{output_folder}/low_res', exist_ok=True)
        low_image_path = f'{output_folder}/low_res/{viewpoint_name}.exr'
        mi.util.write_bitmap(low_image_path, v_low_image)

        os.makedirs(f'{output_folder}/predictions', exist_ok=True)
        image_path = f'{output_folder}/predictions/{viewpoint_name}.exr'
        mi.util.write_bitmap(image_path, image)
        print(f' -- Predicted image has been saved into: {image_path}')

        os.makedirs(f'{output_folder}/references', exist_ok=True)
        ref_image_path = f'{output_folder}/references/{viewpoint_name}.exr'
        mi.util.write_bitmap(ref_image_path, v_ref_image)
        print(f' -- Reference image has been saved into: {ref_image_path}')

        predictions.append(image_path)
        references.append(ref_image_path)
        low_res_images.append(low_image_path)


    images_path = list(zip(predictions, references, low_res_images))

    fig, axs = plt.subplots(len(images_path), 3, figsize=(12, 3 * len(images_path)))

    for p_i, (pred_path, ref_path, low_path) in enumerate(images_path):

        viewpoint_name = viewpoints[p_i]

        pred_image = np.asarray(cv2.imread(pred_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        ref_image = np.asarray(cv2.imread(ref_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        low_image = np.asarray(cv2.imread(low_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))

        gnn_ssim_score = SSIM(pred_image, ref_image, channel_axis=2, data_range=255)
        gnn_mse_score = MSE(pred_image, ref_image)

        low_ssim_score = SSIM(low_image, ref_image, channel_axis=2, data_range=255)
        low_mse_score = MSE(low_image, ref_image)

        axs[p_i, 0].imshow(low_image)
        axs[p_i, 0].set_title(f'From: (SSIM: {low_ssim_score:.4f}, MSE: {low_mse_score:.4f})')
        axs[p_i, 0].axis('off')

        axs[p_i, 1].imshow(pred_image)
        axs[p_i, 1].set_title(f'GNN: (SSIM: {gnn_ssim_score:.4f}, MSE: {gnn_mse_score:.4f})')
        axs[p_i, 1].axis('off')

        axs[p_i, 2].imshow(ref_image)
        axs[p_i, 2].set_title(f'Reference ({viewpoint_name})')
        axs[p_i, 2].axis('off')

    plt.savefig(f'{output_folder}/report.pdf')

if __name__ == "__main__":
    main()
