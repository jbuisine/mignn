import os
import argparse
import dill
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

import mitsuba as mi
mi.set_variant('scalar_rgb')

# ignore Drjit warning
import warnings
warnings.filterwarnings('ignore')

from mignn.container import SimpleLightGraphContainer
from mignn.manager import LightGraphManager


def main():

    parser = argparse.ArgumentParser(description="Simple script only use for building connections")
    parser.add_argument('--gnn_file', type=str, help="gnn file data", required=True)
    parser.add_argument('--scene', type=str, help="mitsuba xml scene", required=True)
    parser.add_argument('--reference', type=str, help="path of the reference image", required=True)
    parser.add_argument('--output', type=str, help="output built container", required=True)

    args = parser.parse_args()

    gnn_file         = args.gnn_file
    scene_file       = args.scene
    image_path       = args.reference
    output_container = args.output

    ref_image = np.asarray(cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))

    container = SimpleLightGraphContainer.fromfile(gnn_file, scene_file, ref_image, verbose=False)

    # TODO: split container here in order to manage huge volume of data?
    # Or try to doing this before? By Splitting and cropping

    # build connections into container and stack graphs
    container.build_connections(n_graphs=10, n_nodes_per_graphs=5, n_neighbors=5, verbose=True)
    build_container = LightGraphManager.vstack(container)

    # save new obtained container
    folder, _ = os.path.split(output_container)
    os.makedirs(folder, exist_ok=True)

    outfile = open(output_container, 'wb')
    dill.dump(build_container, outfile)

if __name__ == "__main__":
    main()
