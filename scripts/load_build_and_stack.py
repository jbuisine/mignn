import os
import argparse
import uuid
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

import mitsuba as mi
mi.set_variant('scalar_rgb')

# ignore Drjit warning
import warnings
warnings.filterwarnings('ignore')

from mignn.dataset import PathLightDataset
from mignn.container import SimpleLightGraphContainer
from mignn.manager import LightGraphManager

from config import N_GRAPHS, N_NODES_PER_GRAPHS, N_NEIGHBORS

def main():

    parser = argparse.ArgumentParser(description="Simple script only use for building connections")
    parser.add_argument('--gnn_folder', type=str, help="gnn file data", required=True)
    parser.add_argument('--scene', type=str, help="mitsuba xml scene", required=True)
    parser.add_argument('--reference', type=str, help="path of the reference image", required=True)
    parser.add_argument('--output', type=str, help="output built container", required=True)

    args = parser.parse_args()

    gnn_folder       = args.gnn_folder
    scene_file       = args.scene
    image_path       = args.reference
    output_dataset   = args.output

    ref_image = np.asarray(cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))

    os.makedirs(output_dataset, exist_ok=True)
    
    for gnn_file_name in sorted(os.listdir(gnn_folder)):
        
        gnn_file_path = os.path.join(gnn_folder, gnn_file_name)
        current_container = SimpleLightGraphContainer.fromfile(gnn_file_path, scene_file, ref_image, verbose=False)
            
        current_container.build_connections(n_graphs=N_GRAPHS, 
                                n_nodes_per_graphs=N_NODES_PER_GRAPHS, 
                                n_neighbors=N_NEIGHBORS, 
                                verbose=False)
        build_container = LightGraphManager.vstack(current_container, verbose=False)

        # save intermediate dataset path
        dataset_path = os.path.join(output_dataset, f'{str(uuid.uuid4())}.path')
            
        PathLightDataset.from_container(build_container, dataset_path, load=False, verbose=False)
        del current_container
        del build_container

if __name__ == "__main__":
    main()
