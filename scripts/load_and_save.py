import os
import argparse
import uuid

# ignore Drjit warning
import warnings
warnings.filterwarnings('ignore')

from utils import load_and_convert
from mignn.dataset import PathLightDataset

def main():

    parser = argparse.ArgumentParser(description="Simple script only use for loading generated data")
    parser.add_argument('--gnn_file', type=str, help="gnn pack file", required=True)
    parser.add_argument('--output', type=str, help="output built container", required=True)

    args = parser.parse_args()

    gnn_filepath   = args.gnn_file
    output_dataset = args.output

    os.makedirs(output_dataset, exist_ok=True)
    
    # save intermediate dataset path
    sub_dataset_path = os.path.join(output_dataset, f'{str(uuid.uuid4())}.pack')
    
    # read graphs data from this subset
    graphs_data = load_and_convert(gnn_filepath)
    
    # TODO: keep only valid graphs (need to check images: during rendering avoid use of empty graph too)
    kept_graphs = []
    for graph in graphs_data:
        if graph.x.size()[0] > 0:
            kept_graphs.append(graph)
    
    # save dataset
    PathLightDataset(sub_dataset_path, kept_graphs)

if __name__ == "__main__":
    main()
