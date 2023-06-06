import os
import argparse
import json
import numpy as np
from itertools import chain

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from mignn.dataset import PathLightDataset

import torch
from joblib import load as skload

from models.gcn_model import GNNL
from models.nerf import BasicNeRf

from utils import prepare_data, scale_subset, merge_by_chunk
from utils import load_sensor_from, load_and_save
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
    parser.add_argument('--scalers', type=str, help="where to find data scalers", required=True)
    parser.add_argument('--model', type=str, help="where to find saved model", required=True)
    parser.add_argument('--output', type=str, help="output data folder", required=True)
    parser.add_argument('--predictions', type=str, help="output predictions folder", required=True)
    parser.add_argument('--sensors', type=str, help="specific sensors folder", required=True)
    
    args = parser.parse_args()

    scene_file         = args.scene
    scalers_folder     = args.scalers
    model_folder       = args.model
    output_folder      = args.output
    predictions_folder = args.predictions
    sensors_folder     = args.sensors
    
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
        sensor = load_sensor_from((w_size, h_size),
                                sensor_file=file_path,
                                integrator=MIGNNConf.INTEGRATOR, 
                                gnn_until=MIGNNConf.GNN_SPP, 
                                gnn_nodes=MIGNNConf.N_NODES_PER_GRAPHS, 
                                gnn_neighbors=MIGNNConf.N_NEIGHBORS)
        
        sensors.append(sensor)
        viewpoints.append(viewpoint_name)


    predictions = []
    references = []
    low_res_images = []

    gnn_folders = prepare_data(scene_file,
                viewpoints=viewpoints,
                integrator = MIGNNConf.INTEGRATOR,
                max_depth = MIGNNConf.MAX_DEPTH,
                ref_spp = MIGNNConf.REF_SPP,
                sensors = sensors,
                output_folder = f'{output_folder}/generated')

    output_temp = f'{output_folder}/datasets/temp/'
    output_temp_scaled = f'{output_folder}/datasets/temp_scaled/'
            
    print('[Processing] extract generated data for each viewpoint')
    
    if not os.path.exists(output_temp):
        os.makedirs(output_temp, exist_ok=True)

        datasets_params = list(chain.from_iterable([ 
                [ 
                    (
                        os.path.join(v_name, v_subset),
                        os.path.join(output_temp, viewpoints[v_i])
                    )
                    # need to sort file in order to preserve pixels order
                    for v_subset in sorted(os.listdir(v_name))
                ] 
                for v_i, v_name in enumerate(sorted(gnn_folders))
            ]))
        
        # multi-process scale of dataset
        pool_obj_scaled = ThreadPool()

        intermediate_datasets_path = []
        for result in tqdm.tqdm(pool_obj_scaled.imap(load_and_save, datasets_params), total=len(datasets_params)):
            intermediate_datasets_path.append(result)
            
    print('[Processing] scaling all subsets using saved model scalers')
    
    # reload scalers        
    scalers = skload(f'{scalers_folder}/scalers.bin')            

    transforms_list = [ScalerTransform(scalers)]
    
    if MIGNNConf.ENCODING_SIZE is not None:
        transforms_list.append(SignalEncoder(MIGNNConf.ENCODING_SIZE, MIGNNConf.ENCODING_MASK))

    applied_transforms = GeoT.Compose(transforms_list)    
        
    if not os.path.exists(output_temp_scaled):
        os.makedirs(output_temp_scaled, exist_ok=True)

        scaled_params = list(chain.from_iterable([ 
                [ 
                    (
                        os.path.join(output_temp, viewpoints[v_i], v_subset), 
                        scalers_folder,
                        os.path.join(output_temp_scaled, viewpoints[v_i])
                    )
                    # need to sort file in order to preserve pixels order
                    for v_subset in sorted(os.listdir(os.path.join(output_temp, v_name)))
                ] 
                for v_i, v_name in enumerate(sorted(os.listdir(output_temp)))
            ]))
        
        # multi-process scale of dataset
        pool_obj_scaled = ThreadPool()

        intermediate_scaled_datasets_path = []
        for result in tqdm.tqdm(pool_obj_scaled.imap(scale_subset, scaled_params), total=len(scaled_params)):
            intermediate_scaled_datasets_path.append(result)
        
    print('[Processing] prepare chunked datasets for each viewpoint')

    datasets_path = []
    for v_i, viewpoint in enumerate(viewpoints):
        
        # split there into memory chunked datasets
        scaled_path = os.path.join(output_temp_scaled, viewpoint)
        scaled_subsets = sorted([ os.path.join(scaled_path, p) for p in os.listdir(scaled_path) ])
        c_output_folder = f'{output_folder}/datasets/chuncks/{viewpoint}_chunks'
        
        # chunk subsets
        if not os.path.exists(c_output_folder):
            merge_by_chunk(viewpoint, scaled_subsets, c_output_folder, applied_transforms)
        
        datasets_path.append(c_output_folder)
    
    #print('[Cleaning] clear intermediated saved containers')
    #os.system(f'rm -r {output_temp}')
    #os.system(f'rm -r {output_temp_scaled}')
    os.makedirs(predictions_folder, exist_ok=True)
    
    for v_i, sensor in enumerate(sensors):

        dataset_path = datasets_path[v_i]
        viewpoint_name = viewpoints[v_i]

        print(f'[Prediction for viewpoint n°{v_i}: {viewpoint_name}]')

        dataset_info = json.load(open(f'{dataset_path}/metadata', 'r', encoding='utf-8'))
        n_node_features = int(dataset_info['n_node_features'])
        
        viewpoint_dataset_paths = sorted([ os.path.join(dataset_path, p) for p in os.listdir(dataset_path) \
                    if 'metadata' not in p ])

        # Load MODEL
        gnn_model = GNNL(hidden_channels=MIGNNConf.HIDDEN_CHANNELS, n_features=n_node_features)
        
        enc_mask, enc_size = MIGNNConf.ENCODING_MASK, MIGNNConf.ENCODING_SIZE
        nerf_features = sum(enc_mask['origin']) * enc_size * 2 + sum(enc_mask['origin']) \
            + sum(enc_mask['direction']) * enc_size * 2 + sum(enc_mask['direction'])
            
        nerf_model = BasicNeRf(nerf_features, MIGNNConf.NERF_LAYER_SIZE, MIGNNConf.NERF_HIDDEN_LAYERS).to(device)

        gnn_model.load_state_dict(torch.load(f'{model_folder}/model_gnn.pt'))
        gnn_model.eval()
        
        nerf_model.load_state_dict(torch.load(f'{model_folder}/model_nerf.pt'))
        nerf_model.eval()
        
        pred_image = np.empty((h_size, w_size, 3)).astype("float32")
        target_image = np.empty((h_size, w_size, 3)).astype("float32")
        input_image = np.empty((h_size, w_size, 3)).astype("float32")

        n_predict = 0
        n_predictions = int(dataset_info["n_samples"])
        for c_dataset_path in viewpoint_dataset_paths:

            dataset = PathLightDataset(root=c_dataset_path)
            dataset_elements = len(dataset)
            for d_i in range(dataset_elements): 
                
                data = dataset[d_i]
                prediction = model(data.x, data.edge_attr, data.edge_index, batch=data.batch).detach().numpy()
                
                # only if scaler is enabled
                # TODO: take care of encoded output! (cannot use mask when inverse transform)
                # Radiance must be the 3 thirds features to predict
                y_target = data.y_direct.detach().numpy() + data.y_indirect.detach().numpy()
                if scalers.get_scalers_from_field('y') is not None:
                    prediction = scalers.inverse_transform_field('y', prediction)
                    y_target = scalers.inverse_transform_field('y', y_target)
                    
                # pixel coordinate
                h, w = data.pixel
                
                input_image[h, w] = data.radiance.detach().numpy()
                pred_image[h, w] = prediction
                target_image[h, w] = y_target

                print(f' -- Prediction progress: {(n_predict + 1) / n_predictions * 100.:.2f}%', end='\r')
                n_predict += 1

        os.makedirs(f'{predictions_folder}/low_res', exist_ok=True)
        low_image_path = f'{predictions_folder}/low_res/{viewpoint_name}.exr'
        cv2.imwrite(low_image_path, input_image)

        os.makedirs(f'{predictions_folder}/predictions', exist_ok=True)
        image_path = f'{predictions_folder}/predictions/{viewpoint_name}.exr'
        cv2.imwrite(image_path, pred_image)
        print(f' -- Predicted image has been saved into: {image_path}')

        os.makedirs(f'{predictions_folder}/references', exist_ok=True)
        ref_image_path = f'{predictions_folder}/references/{viewpoint_name}.exr'
        cv2.imwrite(ref_image_path, target_image)
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

        max_range = np.max([np.max(pred_image), np.max(ref_image), np.max(low_image)])
        
        gnn_ssim_score = SSIM(pred_image, ref_image, channel_axis=2, data_range=max_range)
        gnn_mse_score = MSE(pred_image, ref_image)

        low_ssim_score = SSIM(low_image, ref_image, channel_axis=2, data_range=max_range)
        low_mse_score = MSE(low_image, ref_image)

        # TODO: display png image instead of EXR (error when displaying)
        im_gamma_correct = np.clip(np.power(low_image, 0.8), 0, 1)
        low_im_fixed = Image.fromarray(np.uint8(im_gamma_correct * 255))
        axs[p_i, 0].imshow(low_im_fixed)
        axs[p_i, 0].set_title(f'From: (SSIM: {low_ssim_score:.4f}, MSE: {low_mse_score:.4f})')
        axs[p_i, 0].axis('off')

        im_gamma_correct = np.clip(np.power(pred_image, 0.8), 0, 1)
        pred_im_fixed = Image.fromarray(np.uint8(im_gamma_correct * 255))
        axs[p_i, 1].imshow(pred_im_fixed)
        axs[p_i, 1].set_title(f'GNN: (SSIM: {gnn_ssim_score:.4f}, MSE: {gnn_mse_score:.4f})')
        axs[p_i, 1].axis('off')

        im_gamma_correct = np.clip(np.power(ref_image, 0.8), 0, 1)
        ref_im_fixed = Image.fromarray(np.uint8(im_gamma_correct * 255))
        axs[p_i, 2].imshow(ref_im_fixed)
        axs[p_i, 2].set_title(f'Reference ({viewpoint_name})')
        axs[p_i, 2].axis('off')

    print(f'[Information] pdf report saved into `{predictions_folder}`')
    plt.savefig(f'{predictions_folder}/report.pdf')

if __name__ == "__main__":
    main()
