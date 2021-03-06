import multiprocessing as mp
import threading
import os
import logging
import tempfile
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from config import \
    CKPT_DIR, CHESSBOARD_SIZE, EVAL_FREQ, \
    EVAL_CPUCT, EVAL_NUM_SIMS, EVAL_MCTS_BATCH, \
    TRAIN_LR
from resnet import load_ckpt
from mcts import MCTS
from gobang_utils import config_log, action_from_prob, mcts_nn_policy_generator


def update_best_ckpt_idx(new_best):
    logging.info("updating best index to {}".format(new_best))
    path = tempfile.mktemp()
    with open(path, "w") as f:
        f.write(str(new_best))
    shutil.move(path, os.path.join(CKPT_DIR, "best"))


class GobangSelfPlayDataset(Dataset):
    def __init__(self, records, num_games):
        self.records = records
        self.num_games = num_games
        self._augument()
        for record in self.records:
            for name in ["chessboard", "p"]:
                record[name] = record[name].copy()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def _augument(self):
        ret = []
        for record in self.records:
            chessboard, p, v = record["chessboard"], record["p"], record["v"]
            for option in range(8):
                new_chessboard = chessboard
                new_p = p
                if (option & 1) > 0:
                    new_chessboard = np.flip(new_chessboard, -1)
                    new_p = np.flip(new_p, -1)
                rot_idx = option >> 1
                new_chessboard = np.rot90(new_chessboard, rot_idx, (-2, -1))
                new_p = np.rot90(new_p, rot_idx, (-2, -1))
                ret.append({"chessboard": new_chessboard, "p": new_p, "v": v})
        np.random.shuffle(ret)
        self.records = ret


class RecordBuffer:
    def __init__(self):
        self.records = []
        self.num_games = 0
        self.cv = threading.Condition()

    def extend(self, records):
        self.cv.acquire()
        self.records.extend(records)
        self.num_games += 1
        self.cv.notify_all()
        self.cv.release()

    def get_dataset(self) -> GobangSelfPlayDataset:
        self.cv.acquire()
        while len(self.records) == 0:
            self.cv.wait()
        records = self.records
        num_games = self.num_games
        self.records = []
        self.num_games = 0
        self.cv.release()
        return GobangSelfPlayDataset(records, num_games)


def get_data_loop(record_buffer: RecordBuffer, data_queue: mp.Queue):
    while True:
        records = data_queue.get()
        record_buffer.extend(records)


def evaluate_against_best_ckpt(candidate_network, device_id) -> bool:
    with open(os.path.join(CKPT_DIR, "best"), "r") as f:
        best_idx = int(f.read())
    best_network = load_ckpt(
        os.path.join(CKPT_DIR, "{}.pt".format(best_idx)), device_id
    )
    best_network.eval()
    candidate_network.eval()

    policies = list(map(
        lambda network: mcts_nn_policy_generator(network, device_id),
        [best_network, candidate_network]
    ))

    who = 0
    chessboard = np.zeros((2, CHESSBOARD_SIZE, CHESSBOARD_SIZE))\
        .astype(np.float32)
    while True:
        t = MCTS(
            chessboard if who == 0 else chessboard[::-1, :, :],
            1, EVAL_MCTS_BATCH,
            policies[who]
        )
        if t.terminated():
            return who == 0
        t.search(EVAL_NUM_SIMS, EVAL_CPUCT, None)
        pi = t.get_pi(0)
        x, y = action_from_prob(pi)
        chessboard[who][x][y] = 1
        who = 1 - who

    return False


def train_main(device_id: str, init_ckpt_idx: int, data_queue: mp.Queue, pid: mp.Value):
    # double fork
    fork_pid = os.fork()
    if fork_pid != 0:
        pid.value = fork_pid
        return

    config_log("train-{}.log".format(os.getpid()))
    record_buffer = RecordBuffer()

    get_data_loop_thread = threading.Thread(
        target=get_data_loop,
        args=(record_buffer, data_queue)
    )
    get_data_loop_thread.start()

    network = load_ckpt(
        os.path.join(CKPT_DIR, "{}.pt".format(init_ckpt_idx)),
        device_id
    )
    logging.info("ckpt #{} has been loaded".format(init_ckpt_idx))

    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=TRAIN_LR,
        weight_decay=1e-4
    )

    last_ckpt_idx = 0
    ckpt_idx = init_ckpt_idx
    while True:
        dataset = record_buffer.get_dataset()
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        network.train()
        for batch_idx, batch in enumerate(data_loader):
            chessboard = batch["chessboard"].to(device_id)
            p = batch["p"].to(device_id)
            v = batch["v"].to(device_id)
            logging.info("batch #{}, size = {}".format(batch_idx, v.size(0)))

            optimizer.zero_grad()
            out_p, out_v = network(chessboard)

            loss = F.mse_loss(v, out_v) - \
                torch.mean(torch.sum(
                    F.log_softmax(out_p.view((-1, CHESSBOARD_SIZE ** 2)), dim=-1) *
                    p.view((-1, CHESSBOARD_SIZE ** 2)),
                    dim=1
                ))

            loss.backward()
            optimizer.step()

        ckpt_idx += dataset.num_games
        logging.info("ckpt #{} has been trained".format(ckpt_idx))
        if ckpt_idx - last_ckpt_idx >= EVAL_FREQ:
            last_ckpt_idx = ckpt_idx
            logging.info(
                "evaluating ckpt #{} against best ckpt".format(ckpt_idx))
            if evaluate_against_best_ckpt(network, device_id):
                torch.save(
                    network.state_dict(),
                    os.path.join(CKPT_DIR, "{}.pt".format(ckpt_idx))
                )
                update_best_ckpt_idx(ckpt_idx)
            else:
                logging.info("fail to win the best ckpt")
