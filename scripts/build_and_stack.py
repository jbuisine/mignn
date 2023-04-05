import os
import argparse
import dill

import mitsuba as mi
mi.set_variant('scalar_rgb')

# ignore Drjit warning
import warnings
warnings.filterwarnings('ignore')


from mignn.manager import LightGraphManager


def main():
    
    parser = argparse.ArgumentParser(description="Simple script only use for building connections")
    parser.add_argument('--container', type=str, help="dumped contained path", required=True)
    parser.add_argument('--output', type=str, help="output built container", required=True)
    
    args = parser.parse_args()
    
    container_path   = args.container
    output_container = args.output
    
    # load container
    container = dill.load(open(container_path, 'rb'))
    
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
