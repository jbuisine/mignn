import os
import argparse
import shutil
from itertools import chain

import tqdm
from multiprocessing.pool import ThreadPool

from utils import prepare_data
from utils import load_sensor_from, load_and_save


import config as MIGNNConf

def main():

    parser = argparse.ArgumentParser(description="Generate GNN data from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--output', type=str, help="output data folder", required=True)
    parser.add_argument('--sensors', type=str, help="file with all viewpoints on scene", required=True)
    parser.add_argument('--mode', type=str, help="train or test mode data generation", choices=['train', 'test'], required=True)
    
    args = parser.parse_args()

    scene_file        = args.scene
    output_folder     = args.output
    sensors_folder    = args.sensors
    mode              = args.mode

    # Some MIGNN params 
    w_size, h_size    = MIGNNConf.VIEWPOINT_SIZE
    sensors_n_samples = MIGNNConf.VIEWPOINT_SAMPLES

    # use of: https://github.com/prise-3d/vpbrt
    # read from camera LookAt folder
    sensors = []
    viewpoints = []
    for v_filename in sorted(os.listdir(sensors_folder)):
        file_path = os.path.join(sensors_folder, v_filename)

        sensor = load_sensor_from((w_size, h_size), 
                                  sensor_file=file_path,
                                  integrator=MIGNNConf.INTEGRATOR, 
                                  gnn_until=MIGNNConf.GNN_SPP, 
                                  gnn_nodes=MIGNNConf.N_NODES_PER_GRAPHS, 
                                  gnn_neighbors=MIGNNConf.N_NEIGHBORS)

        # Use a number of times the same sensors in order to increase knowledge
        # Multiple GNN files will be generated
        for _ in range(sensors_n_samples):
            sensors.append(sensor)
            viewpoints.append(v_filename.replace('.txt', ''))

    # multiple datasets to avoid memory overhead
    output_gnn_data = f'{output_folder}/containers/{mode}'
    output_rendering_data = f'{output_folder}/rendering/{mode}'
    
    if not os.path.exists(output_rendering_data):
        
        print(f'[Data generation: {mode}] start generating GNN data using Mistuba3')
        gnn_folders = prepare_data(scene_file,
                                viewpoints = viewpoints,
                                integrator = MIGNNConf.INTEGRATOR,
                                max_depth = MIGNNConf.MAX_DEPTH,
                                ref_spp = MIGNNConf.REF_SPP,
                                sensors = sensors,
                                output_folder = output_rendering_data)
        
        print(f'\n[Data loading: {mode}] loading generated data from Mistuba3')
        
    if not os.path.exists(output_gnn_data):
        
        gnn_folders = [ os.path.join(output_rendering_data, folder) for folder in os.listdir(output_rendering_data) ]

        # clear previous potential generated data (clean way to generate)
        if os.path.exists(output_gnn_data):
            shutil.rmtree(output_gnn_data)
        
        pool_obj = ThreadPool()

        gnn_params = list(chain(*list([
                [ 
                    (
                        os.path.join(folder, g_file), 
                        os.path.join(output_gnn_data, os.path.split(folder)[-1])
                    ) 
                    for g_file in os.listdir(folder) 
                ] 
                for folder in gnn_folders 
            ])))
        # print(gnn_folders)
        
        # load in parallel same scene file, imply error. Here we load multiple scenes
        # params = list(zip(gnn_files, gnn_folders))

        build_containers = []
        for result in tqdm.tqdm(pool_obj.imap(load_and_save, gnn_params), total=len(gnn_params)):
            build_containers.append(result)
    
    else:
        print(f'[Data generation: {mode}] data already generated')

if __name__ == "__main__":
    main()
