import os
import logging
import glob
import multiprocessing as mp

import torch

from config import SELF_PLAY_PROCS, CKPT_DIR, TRAIN_GPU_IDX
from gobang_utils import config_log
from train import update_best_ckpt_idx, train_main
from resnet import ResNet
from selfplay import self_play_main


if __name__ == "__main__":
    config_log(None)
    if not os.path.isdir(CKPT_DIR):
        logging.info("mkdir {}".format(CKPT_DIR))
        os.mkdir(CKPT_DIR)

    if not os.path.isfile(os.path.join(CKPT_DIR, "best")):
        logging.info("best index not found")

        suffix = "pt"
        ckpts = glob.glob("{}/*.{}".format(CKPT_DIR, suffix))
        ckpts = list(map(
            lambda p: int(os.path.basename(p)[:-(1 + len(suffix))]),
            ckpts
        ))
        ckpts.sort()
        if len(ckpts) == 0:
            logging.info("no ckpt available")
            network = ResNet()
            torch.save(network.state_dict(), os.path.join(CKPT_DIR, "0.pt"))
            logging.info("creating 0.pt as default")
            best_idx = 0
        else:
            best_idx = ckpts[-1]
        update_best_ckpt_idx(best_idx)

    with open(os.path.join(CKPT_DIR, "best"), "r") as f:
        best_idx = int(f.read())

    self_play_procs = []
    data_queue = mp.Queue(1 << 9)
    for gpu_idx in SELF_PLAY_PROCS:
        self_play_procs.append(mp.Process(
            target=self_play_main,
            args=(gpu_idx, data_queue)
        ))
        self_play_procs[-1].start()

    train_proc = mp.Process(
        target=train_main,
        args=(TRAIN_GPU_IDX, best_idx, data_queue)
    )
    train_proc.start()

    # self play and training processes become orphans
