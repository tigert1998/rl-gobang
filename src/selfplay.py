import random
import multiprocessing as mp
import time
import os
import logging

import torch
import torch.nn.functional as F
import numpy as np

from config import \
    CHESSBOARD_SIZE, TRAIN_NUM_SIMS, TRAIN_CPUCT, CKPT_DIR
from mcts import MCTS
from gobang_utils import action_from_prob, config_log, mcts_nn_policy_generator
from resnet import ResNet, load_ckpt


def get_temperature(step):
    if step >= 8:
        return 0
    return 1.0 - step / 8.0


def get_best_ckpt_idx() -> int:
    while True:
        try:
            with open(os.path.join(CKPT_DIR, "best"), "r") as f:
                return int(f.read())
        except:
            logging.warning("cannot get best ckpt index temporarily")
            time.sleep(0.25)


def self_play(gpu_id, network):
    records = []
    t = MCTS(
        0,
        np.zeros((2, CHESSBOARD_SIZE, CHESSBOARD_SIZE)).astype(np.float32),
        mcts_nn_policy_generator(network, gpu_id)
    )
    i = 0
    while not t.terminated():
        with torch.no_grad():
            t.search(TRAIN_NUM_SIMS, TRAIN_CPUCT)
        p = t.get_pi(get_temperature(i))
        records.append({
            "chessboard": t.chessboard(),
            "p": p,
            "v": None
        })
        x, y = action_from_prob(p)
        t.step_forward(x, y)
        i += 1

    records[-1]["v"] = -t.v()
    for i in reversed(range(len(records) - 1)):
        records[i]["v"] = -records[i + 1]["v"]
    return records


def self_play_main(gpu_idx: int, data_queue: mp.Queue):
    config_log("selfplay-{}.log".format(os.getpid()))
    gpu_id = "cuda:{}".format(gpu_idx)

    network = None
    prev_best_idx = None
    while True:
        best_idx = get_best_ckpt_idx()
        if best_idx != prev_best_idx:
            logging.info("found a new best ckpt index: {}".format(best_idx))

            network = load_ckpt(
                os.path.join(CKPT_DIR, "{}.pt".format(best_idx)), gpu_id
            )
            network.eval()

        records = self_play(gpu_id, network)
        logging.info("sending records: len(records) = {}".format(len(records)))
        data_queue.put(records)
        prev_best_idx = best_idx
