cc_library(
    name = "capi",
    srcs = [
        "chessboard.cc",
        "mcts_node.cc",
        "mcts.cc",
        "capi.cc"
    ],
    hdrs = [
        "config.h",
        "chessboard.h",
        "mcts_node.h",
        "mcts.h",
        "static_queue.h"
    ],
    alwayslink=True
)

cc_shared_library(
    name = "capi_shared",
    deps = [
        ":capi",
    ],
)