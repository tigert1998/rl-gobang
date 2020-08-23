from typing import Tuple
import threading
from copy import copy
import random
import itertools
import logging

import numpy as np
import torch
import torch.nn.functional as F

from gobang_utils import \
    stone_is_valid, simple_heuristics, mcts_nn_policy_generator
from config import CHESSBOARD_SIZE
from mcts import MCTS
from resnet import ResNet


class Player:
    def place_stone(self, x, y):
        ...

    def evaluate(self, who, chessboard) -> Tuple[int, int]:
        ...

    def kill(self):
        ...


class HumanPlayer(Player):
    def __init__(self):
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.choice = None
        self.killed = False

    def place_stone(self, x, y):
        self.cv.acquire()
        self.choice = (x, y)
        self.cv.notify_all()
        self.cv.release()

    def evaluate(self, who, chessboard):
        self.cv.acquire()
        self.choice = None
        while not self.killed and (self.choice is None or not stone_is_valid(chessboard, *self.choice)):
            self.cv.wait()
        choice = copy(self.choice)
        self.cv.release()

        return choice

    def kill(self):
        self.cv.acquire()
        self.killed = True
        self.cv.notify_all()
        self.cv.release()


class AIPlayer(Player):
    def __init__(self, policy):
        self.policy = policy

    def evaluate(self, who, chessboard):
        if who == 1:
            chessboard = chessboard[::-1, :, :]
        return self.policy(chessboard)


HUMAN_PLAYER = HumanPlayer()


def _random_policy(chessboard):
    while True:
        x = random.randint(0, CHESSBOARD_SIZE - 1)
        y = random.randint(0, CHESSBOARD_SIZE - 1)
        if stone_is_valid(chessboard, x, y):
            return (x, y)


RANDOM_PLAYER = AIPlayer(_random_policy)


def _basic_mcts_policy(chessboard):
    def base_policy(chessboard):
        n = chessboard.shape[0]
        policy = np.ones((n, CHESSBOARD_SIZE, CHESSBOARD_SIZE)) \
            / CHESSBOARD_SIZE ** 2
        value = np.zeros((n,))
        return policy, value
    t = MCTS(chessboard, 1, 1, base_policy)
    t.search(1600, 3, None)
    pi = t.get_pi(0)
    choices = []
    for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
        if pi[x, y] > 0:
            choices.append((x, y))
    return choices[random.randint(0, len(choices) - 1)]


BASIC_MCTS_PLAYER = AIPlayer(_basic_mcts_policy)


def _greedy_policy(chessboard):
    if not (chessboard.sum() > 0):
        return CHESSBOARD_SIZE // 2, CHESSBOARD_SIZE // 2

    highest = -np.inf
    choice = None
    for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
        if chessboard[:, x, y].sum() > 0:
            continue
        new_chessboard = chessboard.copy()
        new_chessboard[0, x, y] = 1
        v = simple_heuristics(new_chessboard)
        new_chessboard[0, x, y] = 0
        new_chessboard[1, x, y] = 1
        v -= simple_heuristics(new_chessboard)
        if v > highest:
            highest = v
            choice = (x, y)
    return choice


GREEDY_PLAYER = AIPlayer(_greedy_policy)


def _greedy_mcts_policy(chessboard):
    def base_policy(chessboard):
        n = chessboard.shape[0]
        policy = np.ones((n, CHESSBOARD_SIZE, CHESSBOARD_SIZE)) \
            / CHESSBOARD_SIZE ** 2
        value = np.array([simple_heuristics(chessboard[i]) for i in range(n)])
        return policy, value
    t = MCTS(chessboard, 1, 1, base_policy)
    t.search(800, 3, None)
    pi = t.get_pi(0)
    choices = []
    for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
        if pi[x, y] > 0:
            choices.append((x, y))
    return choices[random.randint(0, len(choices) - 1)]


GREEDY_MCTS_PLAYER = AIPlayer(_greedy_mcts_policy)


class NNMCTSAIPlayer(AIPlayer):
    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.network = ResNet()
        self.network.load_state_dict(ckpt)
        self.network.eval()

        base_policy = mcts_nn_policy_generator(self.network, "cpu")

        def policy(chessboard):
            t = MCTS(chessboard, 1, 16, base_policy)
            t.search(1600, 3, None)
            pi = t.get_pi(0)
            choices = []
            for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
                if pi[x, y] > 0:
                    choices.append((x, y))
            return choices[random.randint(0, len(choices) - 1)]

        super().__init__(policy)
