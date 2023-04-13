"""Config training module"""
# generated rendering params 
# scene store into `../notebooks/scenes`
# Must be composed of `viewpoints` and `viewpoints_test` folders
SCENE_NAME            = "teapot" 
REF_SPP               = 10000
GNN_SPP               = 10
MAX_DEPTH             = 16
VIEWPOINT_SIZE        = 16, 16
VIEWPOINT_SAMPLES     = 2

# build connections params
N_GRAPHS              = 10
N_NODES_PER_GRAPHS    = 5
N_NEIGHBORS           = 5

# Input data processing params
ENCODING              = 6 # None means no encoding (by default signal encoding)

# dataset generation and performances
# reduce memory usage while generating dataset
N_CORES_GEN_AND_TRAIN = 15
N_CORES_PREDICT       = 11
VIEWPOINT_CHUNK       = 20 # max size in Mo
DATASET_CHUNK         = 200 # max size in Mo
SCENE_REVERSE         = True # specify if width and height are reversed or not

# Training params
# specific to the server
LOSS                  = 'MSE' # MSE, Huber, MAE are supported 
NORMALIZER            = 'MinMax' # MinMax, Norm, Standard are supported
TRAINING_SPLIT        = 0.8
BATCH_SIZE            = 128
EPOCHS                = 20

# Model params
HIDDEN_CHANNELS       = 256

# Predictions params
PRED_VIEWPOINT_SIZE   = 32, 32
