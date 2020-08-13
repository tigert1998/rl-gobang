import itertools

import torch
import numpy as np

from constants import CHESSBOARD_SIZE
from utils import get_winner


class MCTSNode:
    def __init__(
            self, me, chessboard: np.array,
            network
    ):
        self.me = me
        self.chessboard = chessboard
        self.network = network
        self.use_p_noise = False
        self.p_noise = np.zeros((CHESSBOARD_SIZE, CHESSBOARD_SIZE))

        assert self.me in [0, 1]
        assert self.chessboard.shape == (2, CHESSBOARD_SIZE, CHESSBOARD_SIZE)

        winner = get_winner(self.chessboard)
        self.terminated = winner != -1
        if self.terminated:
            self.v = 0 if winner == -2 else (1 if winner == 0 else -1)
        else:
            self.childs = [
                [None] * CHESSBOARD_SIZE
                for _ in range(CHESSBOARD_SIZE)
            ]
            self.p, self.v = self.network(self.chessboard)

            p_sum = self.p.sum()
            eps = 1e-3
            assert 1 - eps <= p_sum and p_sum <= 1 + eps
            assert -1 <= self.v and self.v <= 1

        self.n = 0
        self.sigma_v = 0

    def _construct_child_chessboard(self, x, y):
        ret = self.chessboard.copy()
        ret = ret[::-1, :, :]
        ret[1, x, y] = 1
        return ret

    def set_p_noise(self, dirichlet_alpha: float):
        self.use_p_noise = True
        self.p_noise = np.random.dirichlet(
            dirichlet_alpha * np.ones((CHESSBOARD_SIZE ** 2,))).reshape(
                (CHESSBOARD_SIZE, CHESSBOARD_SIZE))

    def q(self):
        return self.sigma_v * 1.0 / self.n

    def backup(self, delta_v):
        self.n += 1
        self.sigma_v += delta_v

    def expand(self, x, y):
        assert self.childs[x][y] is None
        self.childs[x][y] = MCTSNode(
            1 - self.me,
            self._construct_child_chessboard(x, y),
            self.network
        )

    def select(self, cpuct):
        ret = None
        highest = -np.inf

        for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
            if self.chessboard[:, x, y].sum() > 0:
                continue
            child = self.childs[x][y]
            if self.use_p_noise:
                e = 0.25
                p = (1 - e) * self.p[x][y] + e * self.p_noise[x][y]
            else:
                p = self.p[x][y]
            if child is None:
                tmp = cpuct * p * (self.n ** 0.5)
            else:
                tmp = -self.childs[x][y].q() +\
                    cpuct * p * (self.n ** 0.5) / (child.n + 1)

            if tmp > highest:
                highest = tmp
                ret = (x, y)

        return ret


class MCTS:
    def __init__(self, me, chessboard, network):
        self.me = me
        self.chessboard = chessboard
        self.network = network
        self.root = None

    @classmethod
    def from_mcts_node(cls, node: MCTSNode):
        t = MCTS(node.me, node.chessboard, node.network)
        t.root = node
        return t

    def search(self, num_sims: int, use_p_noise=False):
        if self.root is None:
            self.root = MCTSNode(self.me, self.chessboard, self.network)

        if use_p_noise:
            dirichlet_alpha = 0.03
            self.root.set_p_noise(dirichlet_alpha)

        for _ in range(num_sims):
            self.simulate()

    def simulate(self):
        node = self.root

        cpuct = 5

        expanded = False
        path = [node]
        while not node.terminated and not expanded:
            x, y = node.select(cpuct)
            if node.childs[x][y] is None:
                node.expand(x, y)
                expanded = True
            node = node.childs[x][y]
            path.append(node)

        delta_v = node.v
        for node in reversed(path):
            node.backup(delta_v)
            delta_v *= -1

    def get_pi(self, temperature) -> np.array:
        pi = np.zeros((CHESSBOARD_SIZE, CHESSBOARD_SIZE))
        deno = 0
        highest = 0
        pos = None
        for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
            child = self.root.childs[x][y]
            if child is None:
                continue
            if temperature == 0:
                if child.n > highest:
                    pos = [(x, y)]
                    highest = child.n
                elif child.n == highest:
                    pos.append((x, y))
            else:
                pi[x, y] = child.n ** (1.0 / temperature)
                deno += pi[x, y]

        if temperature == 0:
            for x, y in pos:
                pi[x, y] = 1.0 / len(pos)
        else:
            pi = pi / deno

        return pi.astype(np.float32)
