"""Config training module"""
# generated rendering params 
REF_SPP             = 1000
GNN_SPP             = 10
MAX_DEPTH           = 16
VIEWPOINT_SIZE      = 8, 8
VIEWPOINT_SAMPLES   = 2

# build connections params
N_GRAPHS            = 10
N_NODES_PER_GRAPHS  = 5
N_NEIGHBORS         = 5

# Input data processing params
ENCODING            = 6 # None means no encoding (by default signal encoding)

# dataset generation
# reduce memory usage while generating dataset
VIEWPOINT_CHUNK     = 20 # max size in Mo
DATASET_CHUNK       = 200 # max size in Mo

# Training params
TRAINING_SPLIT      = 0.8
BATCH_SIZE          = 128
EPOCHS              = 20

# Model params
HIDDEN_CHANNELS     = 256

# Predictions params
PRED_VIEWPOINT_SIZE = 64, 64
