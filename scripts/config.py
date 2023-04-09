"""Config training module"""
# generated rendering params 
REF_SPP             = 1000
GNN_SPP             = 10
MAX_DEPTH           = 5
VIEWPOINT_SIZE      = 16, 16
VIEWPOINT_SAMPLES   = 1

# build connections params
N_GRAPHS            = 10
N_NODES_PER_GRAPHS  = 5
N_NEIGHBORS         = 5

# Input data processing params
ENCODING            = None # None means no encoding (by default signal encoding)

# dataset generation
# reduce memory usage while generating dataset
CHUNK_SIZE          = 10000

# Training params
TRAINING_SPLIT      = 0.8
BATCH_SIZE          = 128
EPOCHS              = 40

# Model params
HIDDEN_CHANNELS     = 256

# Predictions params
PRED_VIEWPOINT_SIZE = 32, 32
