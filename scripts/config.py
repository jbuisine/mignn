"""Config training module"""
# generated rendering params 
# scene store into `../notebooks/scenes`
# Must be composed of `viewpoints` and `viewpoints_test` folders
SCENE_NAME            = "teapot-double" 
INTEGRATOR            = "pathgnn"
REF_SPP               = 1000
GNN_SPP               = 20
MAX_DEPTH             = 5
VIEWPOINT_SIZE        = 64, 64
VIEWPOINT_SAMPLES     = 1

# [Build connections params]
N_NODES_PER_GRAPHS    = 10
N_NEIGHBORS           = 10

# [Input data processing params]
# k means (k x 2) additional features by feature (cos(2^k) + sin(2^k))
# TODO: specific encoding size for each field
ENCODING_SIZE         = 6 # None means no encoding (by default signal encoding)
ENCODING_MASK         = {
    'x_node': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'x_edge': [0],
    'y_direct': [0, 0, 0],
    'y_indirect': [0, 0, 0],
    'origin': [1, 1, 1],
    'direction': [1, 1, 1]
}

# [Dataset generation and performances params]
# reduce memory usage while generating dataset
N_CORES_GEN_AND_TRAIN = 15
N_CORES_PREDICT       = 11
DATASET_CHUNK         = 200 # max size in Mo
SCENE_REVERSE         = True # specify if width and height are reversed or not

# [Training params]
# specific to the server
LOSS                  = 'MSE' # MSE, Huber, MAE are supported 
# (MinMax, Robust, Standard, LogMinMax, LogRobust, LogStandard) normalizers are supported
# `None` for no normalization
# Use of scalers for specific fields
NORMALIZERS           = {
    'x_node': {
        #'MinMax': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        'Standard': [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    },
    'x_edge': {
        'MinMax': [1]
    },
    'y_direct': {
        'Log': [1, 1, 1],
        'Standard': [1, 1, 1]
    },
    'y_indirect': {
        'Log': [1, 1, 1],
        'Standard': [1, 1, 1]
    },
    'origin': {
        'Standard': [1, 1, 1]
    },
    'direction': {
        'Standard': [1, 1, 1]
    }
} 

NERF_LAYER_SIZE       = 256
NERF_HIDDEN_LAYERS    = 6
# percentage of data to keep into train and test subsets (by default all)
# usefull when images are in high resolution
DATASET_PERCENT       = 1
TRAINING_SPLIT        = 0.8
BATCH_SIZE            = 128
EPOCHS                = 20

# [Model params]
HIDDEN_CHANNELS       = 256

# [Predictions params]
PRED_VIEWPOINT_SIZE   = 64, 64
