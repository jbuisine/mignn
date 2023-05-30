import os
import argparse
import shutil
from itertools import chain

import mitsuba as mi
mi.set_variant("scalar_rgb")

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

    # multiple datasets to avoid memory overhead
    output_gnn_data = f'{output_folder}/containers'
    output_rendering_data = f'{output_folder}/rendering'
    
    if not os.path.exists(output_rendering_data):
        
        print('[Data generation] start generating GNN data using Mistuba3')
        gnn_folders = prepare_data(scene_file,
                                    integrator = MIGNNConf.INTEGRATOR,
                                    max_depth = MIGNNConf.MAX_DEPTH,
                                    ref_spp = MIGNNConf.REF_SPP,
                                    sensors = sensors,
                                    output_folder = f'{output_folder}/rendering')
        
        print('\n[Building connections] creating connections using Mistuba3')
        
    if not os.path.exists(output_gnn_data):
        
        gnn_folders = [ os.path.join(output_rendering_data, folder) for folder in os.listdir(output_rendering_data) ]

        # clear previous potential generated data (clean way to generate)
        if os.path.exists(output_gnn_data):
            shutil.rmtree(output_gnn_data)
        
        pool_obj = ThreadPool()

        gnn_files = list(chain(*list([
                [ os.path.join(folder, g_file) for g_file in os.listdir(folder) ] 
                for folder in gnn_folders 
            ])))
        
        # load in parallel same scene file, imply error. Here we load multiple scenes
        params = list(zip(gnn_files,
                    [ output_gnn_data for _ in range(len(gnn_files)) ],
                ))

        build_containers = []
        for result in tqdm.tqdm(pool_obj.imap(load_and_save, params), total=len(params)):
            build_containers.append(result)
    
    else:
        print('[Data generation] data already generated')

if __name__ == "__main__":
    main()
