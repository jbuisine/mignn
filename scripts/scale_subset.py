import os
import argparse

from joblib import load as skload

# ignore Drjit warning
import warnings
warnings.filterwarnings('ignore')

import torch_geometric.transforms as GeoT

from mignn.dataset import PathLightDataset
import config as MIGNNConf

from mignn.processing import ScalerTransform, SignalEncoder

def main():

    parser = argparse.ArgumentParser(description="Simple script only use for scaling subsets in parallel")
    parser.add_argument('--dataset', type=str, help="gnn file data", required=True)
    parser.add_argument('--scalers', type=str, help="where scalers are saved", required=True)
    parser.add_argument('--output', type=str, help="output scaled subset folder", required=True)

    args = parser.parse_args()

    dataset_path     = args.dataset
    scalers_folder   = args.scalers
    output_folder    = args.output
    
    _, dataset_name = os.path.split(dataset_path)
    
    c_dataset = PathLightDataset(root=dataset_path)
    
    c_scaled_dataset_path = os.path.join(output_folder, dataset_name)
    
    # create transforms from MIGNNConf
    scalers = skload(f'{scalers_folder}/scalers.bin')
    
    transforms_list = [ScalerTransform(scalers)]
    
    if MIGNNConf.ENCODING is not None:
        transforms_list.append(SignalEncoder(MIGNNConf.ENCODING, MIGNNConf.MASK))

    applied_transforms = GeoT.Compose(transforms_list) 
    
    PathLightDataset(c_scaled_dataset_path, c_dataset, 
                                    pre_transform=applied_transforms)

if __name__ == "__main__":
    main()
