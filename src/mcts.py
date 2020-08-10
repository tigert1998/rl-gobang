from constants import CHESSBOARD_SIZE
import numpy as np
import itertools


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

        self.childs = [
            [None] * CHESSBOARD_SIZE
            for _ in range(CHESSBOARD_SIZE)
        ]
        p, v = self.network(self.chessboard)
        self.p = p[0, :, :]
        self.v = v[0, 0]
        self.n = 1
        self.sigma_v = self.v
        self.q = self.v

    def _child_n(self, x, y):
        return 0 if (self.childs[x][y] is None) else self.childs[x][y].n

    def _construct_child_chessboard(self, x, y):
        ret = self.chessboard.copy()
        ret = ret[:, ::-1, :, :]
        ret[0, 0, x, y] = 1
        return ret

    def _update(self):
        self.n = 1
        self.sigma_v = self.v
        for x in range(CHESSBOARD_SIZE):
            for y in range(CHESSBOARD_SIZE):
                if self.childs[x][y] is None:
                    continue
                child = self.childs[x][y]
                self.n += child.n
                self.sigma_v += -child.sigma_v
        self.q = self.sigma_v / self.n

    def ucb(self, x, y, cpuct):
        return cpuct * self.p[x][y] * (self.n ** 0.5) / (self._child_n(x, y) + 1)

    def select(self, cpuct):
        choice = None
        highest_bound = 0

        for x in range(CHESSBOARD_SIZE):
            for y in range(CHESSBOARD_SIZE):
                if self.chessboard[0, 0, x, y] > 0 or self.chessboard[0, 1, x, y] > 0:
                    continue
                if self.childs[x][y] is None:
                    self.childs[x][y] = MCTSNode(
                        1 - self.me,
                        self._construct_child_chessboard(x, y),
                        self.network
                    )

        self._update()

        for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
            tmp = self.childs[x][y].q + self.ucb(x, y, cpuct)
            if tmp > highest_bound:
                highest_bound = tmp
                choice = (x, y)

        return choice


class MCTS:
    def __init__(self, me, chessboard, network):
        self.me = me
        self.network = network
        self.root = MCTSNode(me, chessboard, network)

    def search(self, num_sims: int):
        for _ in range(num_sims):
            self.simulate()

    def simulate(self):
        node = self.root
        
        path = [node]
        

        for node in reversed(path):
            node._update()
        
        
