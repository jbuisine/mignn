import os
import argparse
import subprocess

import config as MIGNNConf

def main():
    
    parser = argparse.ArgumentParser(description="Run all: dataset creation, train and predict (based on the config file)")
    parser.add_argument('--output', type=str, help="folder where to save data", required=True)
    parser.add_argument('--model', type=str, help="folder where to save learned model", required=True)
    parser.add_argument('--predictions', type=str, help="folder where to save predictions images/analysis", required=True)

    args = parser.parse_args()

    output_folder      = args.output
    output_model       = args.model
    predictions_folder = args.predictions
    
    os.makedirs(predictions_folder, exist_ok=True)
    
    # save config state
    with open('./config.py', encoding='utf-8') as f_input:
        with open(os.path.join(predictions_folder, 'config.state'), 'w', encoding='utf-8') as f_out:
            f_out.writelines(f_input.readlines())
    
    SCENE_FILE = f'../notebooks/scenes/{MIGNNConf.SCENE_NAME}/scene.xml'
    OUTPUT_DATA = os.path.join(f'{output_folder}', 'data')
            
    OUTPUT_DATASET = os.path.join(f'{output_folder}', 'dataset')
    
    OUTPUT_PREDICT = os.path.join(f'{output_folder}', 'predict')
    
    MODEL_FOLDER = os.path.join(f'{output_model}', 'model')

    TRAIN_VIEWPOINTS = f'../notebooks/scenes/{MIGNNConf.SCENE_NAME}/viewpoints_train'
    TEST_VIEWPOINTS = f'../notebooks/scenes/{MIGNNConf.SCENE_NAME}/viewpoints_test'
    
    print(f'[Information] results will be saved into: `{OUTPUT_DATA}`')    
    
    # run data generation
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES}',
        'python', 'prepare_data.py',
        '--scene', SCENE_FILE,
        '--output', OUTPUT_DATA,
        '--sensors', TRAIN_VIEWPOINTS,
        '--mode', 'train'
    ], check=True)
    
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES}',
        'python', 'prepare_data.py',
        '--scene', SCENE_FILE,
        '--output', OUTPUT_DATA,
        '--sensors', TEST_VIEWPOINTS,
        '--mode', 'test'
    ], check=True)
    
    # Run dataset generation
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES}',
        'python', 'generate_dataset.py',
        '--data', f'{OUTPUT_DATA}/containers',
        '--output', OUTPUT_DATASET,
    ], check=True)
    
    # Run train model
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES}', 
        'python', 'train.py', 
        '--dataset', f'{OUTPUT_DATASET}',
        '--output', MODEL_FOLDER 
    ], check=True)
    
    # # Run predictions from model
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES}', 
        'python', 'predict.py', 
        '--scalers', f'{OUTPUT_DATASET}/scalers',
        '--model', f'{MODEL_FOLDER}/model', 
        '--data', f'{OUTPUT_DATASET}/data/test',
        '--predictions', f'{predictions_folder}_test'
    ], check=True)
    
    subprocess.run([
        'taskset', '--cpu-list', f'0-{MIGNNConf.N_CORES}', 
        'python', 'predict.py', 
        '--scalers', f'{OUTPUT_DATASET}/scalers',
        '--model', f'{MODEL_FOLDER}/model', 
        '--data', f'{OUTPUT_DATASET}/data/train',
        '--predictions', f'{predictions_folder}_train'
    ], check=True)

if __name__ == "__main__":
    main()
