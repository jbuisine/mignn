import os
import argparse
import subprocess

import config as MIGNNConf

def main():
    
    parser = argparse.ArgumentParser(description="Run all: dataset creation, train and predict (based on the config file)")
    parser.add_argument('--output', type=str, help="folder where to save data", required=True)

    args = parser.parse_args()

    output_folder = args.output
    
    scalers = '-'.join([ str(v) for _, v in MIGNNConf.NORMALIZERS.items() ])

    SCENE_FILE = f'../notebooks/scenes/{MIGNNConf.SCENE_NAME}/scene.xml'
    OUTPUT_DATA = os.path.join(f'{output_folder}', f'{MIGNNConf.SCENE_NAME}_' \
        f'IMG_{MIGNNConf.VIEWPOINT_SIZE[0]}_{MIGNNConf.VIEWPOINT_SIZE[1]}_' \
        f'S-G_{MIGNNConf.GNN_SPP}_' \
        f'N-G_{MIGNNConf.N_GRAPHS}_' \
        f'N-NG_{MIGNNConf.N_NODES_PER_GRAPHS}_' \
        f'N-NB_{MIGNNConf.N_NEIGHBORS}_' \
        f'VS_{MIGNNConf.VIEWPOINT_SAMPLES}_' \
        f'ENC_{MIGNNConf.ENCODING}_' \
        f'M-{"".join(list(map(str, MIGNNConf.MASK)))}_' \
        f'D_{MIGNNConf.MAX_DEPTH}_' \
        f'EP_{MIGNNConf.EPOCHS}_' \
        f'BS_{MIGNNConf.BATCH_SIZE}_' \
        f'LOSS_{MIGNNConf.LOSS}_' \
        f'NORM_{scalers}')
    OUTPUT_PREDICT = f'{OUTPUT_DATA}_predict'

    TRAIN_VIEWPOINTS = f'../notebooks/scenes/{MIGNNConf.SCENE_NAME}/viewpoints'
    TEST_VIEWPOINTS = f'../notebooks/scenes/{MIGNNConf.SCENE_NAME}/viewpoints_test'
    
    print(f'[Information] results will be saved into: `{OUTPUT_DATA}`')    
    # Run dataset generation
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES_GEN_AND_TRAIN}',
        'python', 'prepare_dataset.py',
        '--scene', SCENE_FILE,
        '--output', OUTPUT_DATA,
        '--sensors', TRAIN_VIEWPOINTS
    ], check=True)
    
    # Run train model
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES_GEN_AND_TRAIN}', 
        'python', 'train.py', 
        '--data', OUTPUT_DATA 
    ], check=True)
    
    # Run predictions from model
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES_PREDICT}', 
        'python', 'predict.py', 
        '--scene', SCENE_FILE, 
        '--model', f'{OUTPUT_DATA}/model', 
        '--output', OUTPUT_PREDICT, 
        '--sensors', TEST_VIEWPOINTS
    ], check=True)

if __name__ == "__main__":
    main()
