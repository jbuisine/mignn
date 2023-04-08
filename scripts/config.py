"""Config training module"""
# generated rendering params 
REF_SPP            = 1000
GNN_SPP            = 10
MAX_DEPTH          = 5

# build connections params
N_GRAPHS           = 10
N_NODES_PER_GRAPHS = 5
N_NEIGHBORS        = 5

# dataset generation
# reduce memory usage while generating dataset
CHUNK_SIZE         = 10000
