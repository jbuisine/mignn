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

    parser = argparse.ArgumentParser(description="Simple script only use for scaling one subset of viewpoint")
    parser.add_argument('--subset', type=str, help="subset data", required=True)
    parser.add_argument('--scalers', type=str, help="where scalers are saved", required=True)
    parser.add_argument('--output', type=str, help="output temp scaled subset", required=True)

    args = parser.parse_args()
    
    subset_path   = args.subset
    scalers_folder   = args.scalers
    output_folder    = args.output
    
    # load scalers
    scalers = skload(scalers_folder)
        
    transforms_list = [ScalerTransform(scalers)]
    
    if MIGNNConf.ENCODING_SIZE is not None:
        transforms_list.append(SignalEncoder(MIGNNConf.ENCODING_SIZE, MIGNNConf.ENCODING_MASK))
    
    applied_transforms = GeoT.Compose(transforms_list)
    
    # prepare temp scaled datasets
    
    # load dataset and then perform scalers
    c_dataset = PathLightDataset(root=subset_path)
    _, dataset_name = os.path.split(subset_path)
    c_scaled_dataset_path = os.path.join(output_folder, dataset_name)
    
    PathLightDataset(c_scaled_dataset_path, c_dataset, 
                                        pre_transform=applied_transforms)
    
    # perform merge
    # merge_by_chunk(viewpoint_scaled_subsets, output_folder, applied_transforms)

if __name__ == "__main__":
    main()
