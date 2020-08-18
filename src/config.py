import ctypes

# defines the game


def _read_config_from_capi():
    lib = ctypes.CDLL('bazel-bin/mcts/libcapi.so')

    class Config(ctypes.Structure):
        _fields_ = [
            ('chessboard_size', ctypes.c_int),
            ('in_a_row', ctypes.c_int),
        ]

    lib.global_GetConfig.argtypes = []
    lib.global_GetConfig.restype = Config
    ret = lib.global_GetConfig()
    return ret.chessboard_size, ret.in_a_row


CHESSBOARD_SIZE, IN_A_ROW = _read_config_from_capi()

# defines the network
NUM_RESIDUAL_BLOCKS = 3
NUM_FILTERS = 32
VALUE_HEAD_HIDDEN_UNITS = 128

# defines the self playing process
TRAIN_NUM_SIMS = 1000
TRAIN_CPUCT = 3
TRAIN_ALPHA = 0.03

# defines the evaluation process
EVAL_FREQ = 20
EVAL_NUM_SIMS = 1000
EVAL_CPUCT = 3

# path
CKPT_DIR = "ckpts"

# defines the master behaviour
SELF_PLAY_GPU_IDXS = [0, 0, 0]
TRAIN_GPU_IDX = 2
