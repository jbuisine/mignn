import os
import argparse
import json
import numpy as np
from itertools import chain

import mitsuba as mi
from mitsuba import ScalarTransform4f as T
mi.set_variant("scalar_rgb")

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from mignn.dataset import PathLightDataset

import torch
from joblib import load as skload

from models.gcn_model import GNNL

from utils import prepare_data, scale_subset, merge_by_chunk
from utils import load_sensor_from, load_build_and_stack
import matplotlib.pyplot as plt

import tqdm
from multiprocessing.pool import ThreadPool
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM

from PIL import Image

import torch_geometric.transforms as GeoT
from mignn.processing import ScalerTransform, SignalEncoder

import config as MIGNNConf

# ignore Drjit warning
import warnings
warnings.filterwarnings('ignore')

def main():

    parser = argparse.ArgumentParser(description="Train model from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--model', type=str, help="where to find saved model", required=True)
    parser.add_argument('--output', type=str, help="output data folder", required=True)
    parser.add_argument('--sensors', type=str, help="specific sensors folder", required=True)
    
    args = parser.parse_args()

    scene_file        = args.scene
    model_folder      = args.model
    output_folder     = args.output
    sensors_folder    = args.sensors
    
    # MIGNN param
    w_size, h_size    = MIGNNConf.PRED_VIEWPOINT_SIZE
    
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

    gnn_folders, ref_images, low_images = prepare_data(scene_file,
                            max_depth = MIGNNConf.MAX_DEPTH,
                            data_spp = MIGNNConf.GNN_SPP,
                            ref_spp = MIGNNConf.REF_SPP,
                            sensors = sensors,
                            output_folder = f'{output_folder}/generated',
                            sort_chunks=True)

    output_temp = f'{output_folder}/datasets/temp/'
    os.makedirs(output_temp, exist_ok=True)
    
    output_temp_scaled = f'{output_folder}/datasets/temp_scaled/'
    os.makedirs(output_temp_scaled, exist_ok=True)

    # manage for each viewpoint
    params = list(chain.from_iterable([ 
                [ 
                    (
                        os.path.join(folder, g_file),
                        scene_file,
                        os.path.join(output_temp, viewpoints[f_i]),
                        ref_images[f_i]
                    )
                    # need to sort file in order to preserve pixels order
                    for g_file in sorted(os.listdir(folder)) 
                ] 
                for f_i, folder in enumerate(gnn_folders)
            ]))
    
    print('\n[Building connections] creating connections using Mistuba3')
    # multiprocess build of connections
    pool_obj = ThreadPool()

    build_containers = []
    for result in tqdm.tqdm(pool_obj.imap(load_build_and_stack, params), total=len(params)):
        build_containers.append(result)
        
        
    print('[Processing] scaling all subsets using saved model scalers')
    # specify where to find each subsets    
    scalers_folder = f'{model_folder}/scalers'

    scaled_params = list(chain.from_iterable([ 
            [ 
                (
                    os.path.join(output_temp, v_name, v_subset), 
                    scalers_folder,
                    os.path.join(output_temp_scaled, v_name)
                )
                # need to sort file in order to preserve pixels order
                for v_subset in sorted(os.listdir(os.path.join(output_temp, v_name))) 
            ] 
            for v_name in viewpoints
        ]))
    
    # multi-process scale of dataset
    pool_obj_scaled = ThreadPool()

    intermediate_scaled_datasets_path = []
    for result in tqdm.tqdm(pool_obj_scaled.imap(scale_subset, scaled_params), total=len(scaled_params)):
        intermediate_scaled_datasets_path.append(result)
    
    print('[Processing] prepare chunked datasets for each viewpoint')
    
    x_scaler = skload(f'{scalers_folder}/x_node_scaler.bin')
    edge_scaler = skload(f'{scalers_folder}/x_edge_scaler.bin')
    y_scaler = skload(f'{scalers_folder}/y_scaler.bin')
        
    # reload scalers    
    scalers = {
        'x_node': x_scaler,
        'x_edge': edge_scaler,
        'y': y_scaler
    }
    
    transforms_list = [ScalerTransform(scalers)]
    
    if MIGNNConf.ENCODING is not None:
        transforms_list.append(SignalEncoder(MIGNNConf.ENCODING))

    applied_transforms = GeoT.Compose(transforms_list)    

    
    datasets_path = []
    for v_i, viewpoint in enumerate(viewpoints):
        
        # split there into memory chunked datasets
        scaled_path = os.path.join(output_temp_scaled, viewpoint)
        scaled_subsets = [ os.path.join(scaled_path, p) for p in os.listdir(scaled_path) ]
        c_output_folder = f'{output_folder}/datasets/{viewpoint}_chunks'
        merge_by_chunk(viewpoint, scaled_subsets, c_output_folder, applied_transforms)
        
        datasets_path.append(c_output_folder)
      
    print(f'[cleaning] clear intermediated saved containers')
    os.system(f'rm -r {output_temp}')
    os.system(f'rm -r {output_temp_scaled}')
    
    for v_i, sensor in enumerate(sensors):

        dataset_path = datasets_path[v_i]
        viewpoint_name = viewpoints[v_i]
        v_ref_image = np.asarray(cv2.imread(ref_images[v_i], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        v_low_image = np.asarray(cv2.imread(low_images[v_i], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))

        print(f'[Prediction for viewpoint n°{v_i}: {viewpoint_name}]')

        dataset_info = json.load(open(f'{dataset_path}/metadata', 'r', encoding='utf-8'))
        n_node_features = int(dataset_info['n_node_features'])
        
        viewpoint_dataset_paths = sorted([ os.path.join(dataset_path, p) for p in os.listdir(dataset_path) \
                    if 'metadata' not in p ])

        
        model = GNNL(hidden_channels=MIGNNConf.HIDDEN_CHANNELS, n_features=n_node_features)
        # print('[Information] Model has been loaded')

        model.load_state_dict(torch.load(f'{model_folder}/model.pt'))
        model.eval()
        
        pixels = []

        n_predict = 0
        n_predictions = int(dataset_info["n_samples"])
        for c_dataset_path in viewpoint_dataset_paths:

            dataset = PathLightDataset(root=c_dataset_path)
            dataset_elements = len(dataset)
            for d_i in range(dataset_elements): 
                
                data = dataset[d_i]
                prediction = model(data.x, data.edge_attr, data.edge_index, batch=data.batch)
                prediction = y_scaler.inverse_transform(prediction.detach().numpy())
                pixels.append(prediction)

                print(f' -- Prediction progress: {(n_predict + 1) / n_predictions * 100.:.2f}%', end='\r')
                n_predict += 1

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

        # TODO: display png image instead of EXR (error when displaying)
        #im_gamma_correct = np.clip(np.power(low_image, 2), 0, 1)
        #low_im_fixed = Image.fromarray(np.uint8(im_gamma_correct * 255))
        axs[p_i, 0].imshow(low_image)
        axs[p_i, 0].set_title(f'From: (SSIM: {low_ssim_score:.4f}, MSE: {low_mse_score:.4f})')
        axs[p_i, 0].axis('off')

        #im_gamma_correct = np.clip(np.power(pred_image, 0.45), 0, 1)
        #pred_im_fixed = Image.fromarray(np.uint8(im_gamma_correct * 255))
        axs[p_i, 1].imshow(pred_image)
        axs[p_i, 1].set_title(f'GNN: (SSIM: {gnn_ssim_score:.4f}, MSE: {gnn_mse_score:.4f})')
        axs[p_i, 1].axis('off')

        #im_gamma_correct = np.clip(np.power(ref_image, 0.45), 0, 1)
        #ref_im_fixed = Image.fromarray(np.uint8(im_gamma_correct * 255))
        axs[p_i, 2].imshow(ref_image)
        axs[p_i, 2].set_title(f'Reference ({viewpoint_name})')
        axs[p_i, 2].axis('off')

    print(f'[Information] pdf report saved into `{output_folder}`')
    plt.savefig(f'{output_folder}/report.pdf')

if __name__ == "__main__":
    main()
