"""Config training module"""
# generated rendering params 
# scene store into `../notebooks/scenes`
# Must be composed of `viewpoints` and `viewpoints_test` folders
SCENE_NAME            = "teapot-double" 
REF_SPP               = 10000
GNN_SPP               = 10
MAX_DEPTH             = 5
VIEWPOINT_SIZE        = 16, 16
VIEWPOINT_SAMPLES     = 1

# [Build connections params]
N_GRAPHS              = 10
N_NODES_PER_GRAPHS    = 5
N_NEIGHBORS           = 5

# [Input data processing params]
# k means (k x 2) additional features by feature (cos(2^k) + sin(2^k))
ENCODING              = 6 # None means no encoding (by default signal encoding)
MASK                  = {
    'x_node': [1, 1, 1, 1, 1, 1, 0, 0, 0],
    'x_edge': [1, 1, 1],
    'y': [0, 0, 0]
}

# [Dataset generation and performances params]
# reduce memory usage while generating dataset
N_CORES_GEN_AND_TRAIN = 15
N_CORES_PREDICT       = 11
VIEWPOINT_CHUNK       = 20 # max size in Mo
DATASET_CHUNK         = 200 # max size in Mo
SCENE_REVERSE         = True # specify if width and height are reversed or not

# [Training params]
# specific to the server
LOSS                  = 'MSE' # MSE, Huber, MAE are supported 
# (MinMax, Robust, Standard, LogMinMax, LogRobust, LogStandard) normalizers are supported
# `None` for no normalization
NORMALIZERS           = {
    'x_node': 'MinMax',
    'x_edge': 'MinMax',
    'y': 'LogMinMax'
} 
# percentage of data to keep into train and test subsets (by default all)
# usefull when images are in high resolution
DATASET_PERCENT       = 1
TRAINING_SPLIT        = 0.8
BATCH_SIZE            = 128
EPOCHS                = 100

# [Model params]
HIDDEN_CHANNELS       = 256

# [Predictions params]
PRED_VIEWPOINT_SIZE   = 64, 64
