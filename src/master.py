import os
import logging
import glob
import multiprocessing as mp
import subprocess
import json
import signal
import argparse

import torch

from config import SELF_PLAY_DEVICE_IDS, CKPT_DIR, TRAIN_DEVICE_ID
from gobang_utils import config_log
from train import update_best_ckpt_idx, train_main
from resnet import ResNet
from selfplay import self_play_main


def _master_hidden_file():
    return os.path.join(CKPT_DIR, ".master")


def _best_ckpt_idx_file():
    return os.path.join(CKPT_DIR, "best")


def start():
    if os.path.isfile(_master_hidden_file()):
        logging.warning('run "kill" first to terminate background training')
        return

    if not os.path.isdir(CKPT_DIR):
        logging.info("mkdir {}".format(CKPT_DIR))
        os.mkdir(CKPT_DIR)

    if not os.path.isfile(_best_ckpt_idx_file()):
        logging.info("best index not found")

        ckpts = glob.glob("{}/*.pt".format(CKPT_DIR))
        ckpts = list(map(
            lambda p: int(os.path.splitext(os.path.basename(p))[0]),
            ckpts
        ))
        ckpts.sort()
        if len(ckpts) == 0:
            logging.info("no ckpt available in the ckpt directory")
            network = ResNet()
            torch.save(network.state_dict(), os.path.join(CKPT_DIR, "0.pt"))
            logging.info("creating 0.pt as the default best ckpt")
            best_idx = 0
        else:
            best_idx = ckpts[-1]
        update_best_ckpt_idx(best_idx)

    with open(_best_ckpt_idx_file(), "r") as f:
        best_idx = int(f.read())

    pids = []
    self_play_procs = []
    data_queue = mp.Queue(1 << 9)
    for device_id in SELF_PLAY_DEVICE_IDS:
        pids.append(mp.Value('i', 0))
        self_play_procs.append(mp.Process(
            target=self_play_main,
            args=(device_id, data_queue, pids[-1])
        ))
        self_play_procs[-1].start()
        self_play_procs[-1].join()

    pids.append(mp.Value('i', 0))
    train_proc = mp.Process(
        target=train_main,
        args=(TRAIN_DEVICE_ID, best_idx, data_queue, pids[-1])
    )
    train_proc.start()
    train_proc.join()

    for i in range(len(pids)):
        pids[i] = pids[i].value

    with open(_master_hidden_file(), "w") as f:
        json.dump(pids, f)

    # self play and training processes become orphans


def kill():
    if not os.path.isfile(_master_hidden_file()):
        logging.warning("no background training process is found")
        return

    with open(_master_hidden_file(), "r") as f:
        pids = json.load(f)

    logging.info("killing background processes")
    for pid in pids:
        os.kill(pid, signal.SIGKILL)
    os.remove(_master_hidden_file())


if __name__ == "__main__":
    config_log(None)
    parser = argparse.ArgumentParser(description='master')
    parser.add_argument(
        "instruction", help="the instruction to execute",
        choices=["start", "kill"]
    )
    args = parser.parse_args()
    if args.instruction == "start":
        start()
    elif args.instruction == "kill":
        kill()
