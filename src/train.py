import random
import itertools

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from mcts import MCTS
from resnet import ResNet
from constants import CHESSBOARD_SIZE


class GobangSelfPlayDataset(Dataset):
    def __init__(self, network):
        self.network = network
        self.records = []
        self._self_play()
        self._augument()
        for i in self:
            i["chessboard"] = i["chessboard"].copy()
            i["p"] = i["p"].copy()

    def __len__(self):
        return len(self.records)

    @classmethod
    def _action_from_pi(cls, pi):
        tmp = random.uniform(0, 1)
        for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
            if tmp < pi[x][y]:
                return x, y
            tmp -= pi[x][y]
        return CHESSBOARD_SIZE - 1, CHESSBOARD_SIZE - 1

    def _self_play(self):
        def base_policy(chessboard):
            i = torch.from_numpy(np.expand_dims(
                chessboard.astype(np.float32), axis=0))
            x, y = self.network(i)
            return x.data.numpy()[0, :, :], y.data.numpy()[0]

        chessboard = np.zeros((2, CHESSBOARD_SIZE, CHESSBOARD_SIZE))
        t = MCTS(0, chessboard, base_policy)

        self.network.eval()
        i = 0
        while t.root is None or not t.root.terminated:
            with torch.no_grad():
                t.search(1600, i >= 10)

            p = t.get_pi(1 if i < 10 else 0)
            self.records.append({
                "chessboard": t.root.chessboard,
                "p": p,
                "v": None
            })
            x, y = self._action_from_pi(p)
            t = MCTS.from_mcts_node(t.root.childs[x][y])
            i += 1

        self.records[-1]["v"] = -t.root.v

        for i in reversed(range(len(self.records) - 1)):
            self.records[i]["v"] = -self.records[i + 1]["v"]

    def _augument(self):
        ret = []
        for record in self.records:
            chessboard, p, v = record["chessboard"], record["p"], record["v"]
            for option in range(4):
                new_chessboard = chessboard
                new_p = p
                if (option & 1) > 0:
                    new_chessboard = np.flip(new_chessboard, -2)
                    new_p = np.flip(new_p, -2)
                if (option & 2) > 0:
                    new_chessboard = np.flip(new_chessboard, -1)
                    new_p = np.flip(new_p, -1)
                ret.append({"chessboard": new_chessboard, "p": new_p, "v": v})

            for i in range(3):
                new_chessboard = np.rot90(chessboard, i + 1, (-2, -1))
                new_p = np.rot90(p, i + 1, (-2, -1))
                ret.append({"chessboard": new_chessboard, "p": new_p, "v": v})

        np.random.shuffle(ret)
        self.records = ret

    def __getitem__(self, idx):
        return self.records[idx]


def train():
    network = ResNet()

    dataset = GobangSelfPlayDataset(network)
    data_loader = DataLoader(dataset, batch_size=4)

    for i_batch, sample_batched in enumerate(data_loader):
        print(
            i_batch, sample_batched['chessboard'].size(),
            sample_batched['p'].size(),
            sample_batched["v"]
        )


if __name__ == "__main__":
    train()
