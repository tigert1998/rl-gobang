import itertools
import numpy as np

from constants import CHESSBOARD_SIZE
from gobang_env import get_winner


class MCTSNode:
    def __init__(
            self, me, chessboard: np.array,
            network
    ):
        self.me = me
        self.chessboard = chessboard
        self.network = network

        assert self.me in [0, 1]
        assert self.chessboard.shape == (
            1, 2, CHESSBOARD_SIZE, CHESSBOARD_SIZE
        )

        winner = get_winner(self.chessboard)
        self.terminated = winner != -1
        if self.terminated:
            self.v = 0 if winner == -2 else (1 if winner == 0 else -1)
        else:
            self.childs = [
                [None] * CHESSBOARD_SIZE
                for _ in range(CHESSBOARD_SIZE)
            ]
            p, v = self.network(self.chessboard)
            self.p = p[0, :, :]
            self.v = v[0, 0]

        self.n = 0
        self.sigma_v = 0

    def _construct_child_chessboard(self, x, y):
        ret = self.chessboard.copy()
        ret = ret[:, ::-1, :, :]
        ret[0, 0, x, y] = 1
        return ret

    def q(self):
        return self.sigma_v / self.n

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
            child = self.childs[x][y]
            if child is None:
                tmp = cpuct * self.p[x][y] * (self.n ** 0.5)
            else:
                tmp = -self.childs[x][y].q() +\
                    cpuct * self.p[x][y] * (self.n ** 0.5) / (child.n + 1)

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

    def search(self, num_sims: int):
        if self.root is None:
            self.root = MCTSNode(self.me, self.chessboard, self.network)
        for _ in range(num_sims):
            self.simulate()

    def simulate(self):
        node = self.root

        cpuct = 1

        path = [node]
        while not node.terminated:
            x, y = node.select(cpuct)
            if node.childs[x][y] is None:
                node.expand(x, y)
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
                pi[x, y] = child.n ** (1 / temperature)
                deno += pi[x, y]

        if temperature == 0:
            for x, y in pos:
                pi[x, y] = 1.0 / len(pos)
        else:
            pi = pi / deno

        return pi
