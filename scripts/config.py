"""Config training module"""
# generated rendering params 
# scene store into `../notebooks/scenes`
# Must be composed of `viewpoints_train` and `viewpoints_test` folders
SCENE_NAME            = "living-room-reduced" 
INTEGRATOR            = "pathgnn"
REF_SPP               = 1000
GNN_SPP               = 20
MAX_DEPTH             = 5
VIEWPOINT_SIZE        = 128, 128
VIEWPOINT_SAMPLES     = 1

# [Build connections params]
N_NODES_PER_GRAPHS    = 20
N_NEIGHBORS           = 20

# [Dataset generation and performances params]
# reduce memory usage while generating dataset
N_CORES               = 15
DATASET_CHUNK         = 200 # max size in Mo

# [Model params]
# NeRF: {simple: SimpleNeRF}
# GNN: {simple: GNNL, simple_camera: GNNL_VP, simple_viewpoint: GNNL_VPP}
MODELS =              {
    'nerf': 'simple',
    'gnn': 'simple'
}

LOSS                  = {
    'nerf': 'MSE',
    'gnn': 'MSE'
}

# available modes: {"simple", "separated"}
# simple: GNN needs to predict the whole radiance
# separated: NeRF predicts the direct radiance, GNN the indirect radiance
TRAINING_MODE         = 'simple'

# some specific model params
GNN_HIDDEN_CHANNELS   = 256
GNN_LATENT_SPACE      = 100
GNN_DENSE_HIDDEN      = 256
GNN_N_DENSE_LAYERS    = 4

NERF_LAYER_SIZE       = 256
NERF_HIDDEN_LAYERS    = 6

# [Preprocessing params]
# (MinMax, Robust, Standard, LogMinMax, LogRobust, LogStandard) normalizers are supported
# Use of scalers for specific fields using mask (before signal encoding)
NORMALIZERS           = {
    'x_node': {
        'Log': [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        'MinMax': [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    },
    'x_edge': {
        'MinMax': [1]
    },
    'y_total': {
        'Log': [1, 1, 1],
        'MinMax': [1, 1, 1]
    },
    'y_direct': {
        'Log': [1, 1, 1],
        'MinMax': [1, 1, 1]
    },
    'y_indirect': {
        'Log': [1, 1, 1],
        'MinMax': [1, 1, 1]
    },
    'origin': {
        'MinMax': [1, 1, 1]
    },
    'direction': {
        'MinMax': [1, 1, 1]
    }
} 

# [Input data processing params]
# Encoding is applied after normalization
# WARNING: predicted fields must not be encoded (reverse operation is not possible)
# k means (k x 2) additional features by feature (cos(2^k) + sin(2^k))
# TODO: specific encoding size for each field
ENCODING_SIZE         = 6 # None means no encoding (by default signal encoding)
ENCODING_MASK         = {
    'x_node': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'x_edge': [0],
    'y_total': [0, 0, 0],
    'y_direct': [0, 0, 0],
    'y_indirect': [0, 0, 0],
    'origin': [1, 1, 1],
    'direction': [1, 1, 1]
}


# [Training params]
# percentage of data to keep into train and test subsets (by default all)
# usefull when images are in high resolution
DATASET_PERCENT       = 1
BATCH_SIZE            = 128
EPOCHS                = 20

# [Predictions params]
PRED_VIEWPOINT_SIZE   = 128, 128
