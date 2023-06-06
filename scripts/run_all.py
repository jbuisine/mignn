import os
import argparse
import subprocess

import config as MIGNNConf

def main():
    
    parser = argparse.ArgumentParser(description="Run all: dataset creation, train and predict (based on the config file)")
    parser.add_argument('--output', type=str, help="folder where to save data", required=True)
    parser.add_argument('--predictions', type=str, help="folder where to save predictions images/analysis", required=True)

    args = parser.parse_args()

    output_folder      = args.output
    predictions_folder = args.predictions
    
    # scalers = '-'.join([ str(v) for _, v in MIGNNConf.NORMALIZERS.items() ])

    SCENE_FILE = f'../notebooks/scenes/{MIGNNConf.SCENE_NAME}/scene.xml'
    OUTPUT_DATA = os.path.join(f'{output_folder}', f'{MIGNNConf.SCENE_NAME}_' \
        f'IMG_{MIGNNConf.VIEWPOINT_SIZE[0]}_{MIGNNConf.VIEWPOINT_SIZE[1]}_' \
        f'S-G_{MIGNNConf.GNN_SPP}_' \
        f'N-NG_{MIGNNConf.N_NODES_PER_GRAPHS}_' \
        f'N-NB_{MIGNNConf.N_NEIGHBORS}_' \
        f'VS_{MIGNNConf.VIEWPOINT_SAMPLES}_' \
        f'D_{MIGNNConf.MAX_DEPTH}')
            
    OUTPUT_DATASET = OUTPUT_DATA + \
        f'_ENC_{MIGNNConf.ENCODING_SIZE}_' \
        f'MX-{"".join(list(map(str, MIGNNConf.ENCODING_MASK["x_node"])))}_' \
        f'ME-{"".join(list(map(str, MIGNNConf.ENCODING_MASK["x_edge"])))}_' \
        f'EP_{MIGNNConf.EPOCHS}_' \
        f'BS_{MIGNNConf.BATCH_SIZE}_' \
    
    MODEL_FOLDER = f'{OUTPUT_DATASET}_{MIGNNConf.HIDDEN_CHANNELS}_model'
        
    OUTPUT_PREDICT = f'{OUTPUT_DATASET}_{MIGNNConf.HIDDEN_CHANNELS}_predict'

    TRAIN_VIEWPOINTS = f'../notebooks/scenes/{MIGNNConf.SCENE_NAME}/viewpoints'
    TEST_VIEWPOINTS = f'../notebooks/scenes/{MIGNNConf.SCENE_NAME}/viewpoints_test'
    
    print(f'[Information] results will be saved into: `{OUTPUT_DATA}`')    
    
    # run data generation
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES_GEN_AND_TRAIN}',
        'python', 'prepare_data.py',
        '--scene', SCENE_FILE,
        '--output', OUTPUT_DATA,
        '--sensors', TRAIN_VIEWPOINTS
    ], check=True)
    
    # Run dataset generation
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES_GEN_AND_TRAIN}',
        'python', 'generate_dataset.py',
        '--data', f'{OUTPUT_DATA}/containers',
        '--output', OUTPUT_DATASET,
    ], check=True)
    
    # Run train model
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES_GEN_AND_TRAIN}', 
        'python', 'train.py', 
        '--dataset', f'{OUTPUT_DATASET}/datasets',
        '--output', MODEL_FOLDER 
    ], check=True)
    
    # Run predictions from model
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES_PREDICT}', 
        'python', 'predict.py', 
        '--scene', SCENE_FILE, 
        '--scalers', f'{OUTPUT_DATASET}/datasets/scalers',
        '--model', f'{MODEL_FOLDER}/model', 
        '--output', OUTPUT_PREDICT,
        '--predictions', predictions_folder, 
        '--sensors', TEST_VIEWPOINTS
    ], check=True)

if __name__ == "__main__":
    main()
