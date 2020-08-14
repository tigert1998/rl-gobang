cc_library(
    name = "config",
    srcs = [],
    hdrs = ["config.h"],
    deps = [],
)

cc_library(
    name = "chessboard",
    srcs = ["chessboard.cc"],
    hdrs = ["chessboard.h"],
    deps = [":config"],
)

cc_library(
    name = "mcts_node",
    srcs = ["mcts_node.cc"],
    hdrs = ["mcts_node.h"],
    deps = [":chessboard"],
)

cc_library(
    name = "mcts",
    srcs = ["mcts.cc"],
    hdrs = ["mcts.h"],
    deps = [":mcts_node"],
)

cc_library(
    name = "capi",
    srcs = ["capi.cc"],
    hdrs = [],
    deps = [":mcts"],
)