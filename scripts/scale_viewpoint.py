import os
import argparse

from joblib import load as skload

# ignore Drjit warning
import warnings
warnings.filterwarnings('ignore')

import torch_geometric.transforms as GeoT

from utils import merge_by_chunk

from mignn.dataset import PathLightDataset
import config as MIGNNConf

from mignn.processing import ScalerTransform, SignalEncoder

def main():

    parser = argparse.ArgumentParser(description="Simple script only use for scaling subsets of viewpoint")
    parser.add_argument('--viewpoint', type=str, help="gnn viewpoint data", required=True)
    parser.add_argument('--scalers', type=str, help="where scalers are saved", required=True)
    parser.add_argument('--temp', type=str, help="output temp scaled subset folder", required=True)
    parser.add_argument('--output', type=str, help="output merge subsets folder", required=True)

    args = parser.parse_args()
    
    viewpoint_path   = args.viewpoint
    scalers_folder   = args.scalers
    temp_folder      = args.temp
    output_folder    = args.output
    
    # load scalers
    scalers = skload(scalers_folder)
        
    transforms_list = [ScalerTransform(scalers)]
    
    if MIGNNConf.ENCODING_SIZE is not None:
        transforms_list.append(SignalEncoder(MIGNNConf.ENCODING_SIZE, MIGNNConf.ENCODING_MASK))
    
    applied_transforms = GeoT.Compose(transforms_list)
    
    # prepare temp scaled datasets
    viewpoint_scaled_subsets = []
    for viewpoint_filename in os.listdir(viewpoint_path):
        
        c_dataset_path = os.path.join(viewpoint_path, viewpoint_filename)
        _, dataset_name = os.path.split(c_dataset_path)
        
        # load dataset and then perform scalers
        c_dataset = PathLightDataset(root=c_dataset_path)
        c_scaled_dataset_path = os.path.join(temp_folder, dataset_name)
        viewpoint_scaled_subsets.append(c_scaled_dataset_path)
        
        PathLightDataset(c_scaled_dataset_path, c_dataset, 
                                        pre_transform=applied_transforms)
    
    # perform merge
    merge_by_chunk(viewpoint_scaled_subsets, output_folder, applied_transforms)

if __name__ == "__main__":
    main()
