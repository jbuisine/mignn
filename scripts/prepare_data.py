import os
import argparse
import shutil
import random
import numpy as np
from itertools import chain

import mitsuba as mi
mi.set_variant("scalar_rgb")

import tqdm
from multiprocessing.pool import ThreadPool

from utils import prepare_data
from utils import load_sensor_from, load_build_and_stack


import config as MIGNNConf

def main():

    parser = argparse.ArgumentParser(description="Generate GNN data from multiple viewpoints")
    parser.add_argument('--scene', type=str, help="mitsuba xml scene file", required=True)
    parser.add_argument('--output', type=str, help="output data folder", required=True)
    parser.add_argument('--sensors', type=str, help="file with all viewpoints on scene", required=True)
    
    args = parser.parse_args()

    scene_file        = args.scene
    output_folder     = args.output
    sensors_folder    = args.sensors

    # Some MIGNN params 
    w_size, h_size    = MIGNNConf.VIEWPOINT_SIZE
    sensors_n_samples = MIGNNConf.VIEWPOINT_SAMPLES

    # use of: https://github.com/prise-3d/vpbrt
    # read from camera LookAt folder
    sensors = []
    for file in sorted(os.listdir(sensors_folder)):
        file_path = os.path.join(sensors_folder, file)

        sensor = load_sensor_from((w_size, h_size), file_path)

        # Use a number of times the same sensors in order to increase knowledge
        # Multiple GNN files will be generated
        for _ in range(sensors_n_samples):
            sensors.append(sensor)

    # multiple datasets to avoid memory overhead
    output_gnn_data = f'{output_folder}/containers'

    if not os.path.exists(output_gnn_data):
        
        os.makedirs(output_gnn_data, exist_ok=True)
        
        print('[Data generation] start generating GNN data using Mistuba3')
        gnn_folders, ref_images, _ = prepare_data(scene_file,
                                    max_depth = MIGNNConf.MAX_DEPTH,
                                    data_spp = MIGNNConf.GNN_SPP,
                                    ref_spp = MIGNNConf.REF_SPP,
                                    sensors = sensors,
                                    output_folder = f'{output_folder}/rendering')

        # associate for each file in gnn_folder, the correct reference image
        gnn_files, references = list(zip(*list(chain.from_iterable(list([ 
                        [ (os.path.join(folder, g_file), ref_images[f_i]) for g_file in os.listdir(folder) ] 
                        for f_i, folder in enumerate(gnn_folders) 
                    ])))))
        
        print('\n[Building connections] creating connections using Mistuba3')
        
        # clear previous potential generated data (clean way to generate)
        if os.path.exists(output_gnn_data):
            shutil.rmtree(output_gnn_data)
        
        pool_obj = ThreadPool()

        # load in parallel same scene file, imply error. Here we load multiple scenes
        params = list(zip(gnn_files,
                    [ scene_file for _ in range(len(gnn_files)) ],
                    [ output_gnn_data for _ in range(len(gnn_files)) ],
                    references
                ))

        build_containers = []
        for result in tqdm.tqdm(pool_obj.imap(load_build_and_stack, params), total=len(params)):
            build_containers.append(result)
    
    else:
        print('[Data generation] GNN data already available')
 
if __name__ == "__main__":
    main()
