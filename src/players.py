from typing import Tuple
import threading
from copy import copy
import random
import itertools
import logging

import numpy as np

from utils import stone_is_valid, simple_heuristics
from constants import CHESSBOARD_SIZE
from mcts import MCTS


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
    def base_policy(_):
        policy = np.ones((CHESSBOARD_SIZE, CHESSBOARD_SIZE)) \
            / CHESSBOARD_SIZE ** 2
        value = 0
        return policy, value
    t = MCTS(0, chessboard, base_policy)
    t.search(1600)
    pi = t.get_pi(0)
    choices = []
    for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
        if pi[x, y] > 0:
            choices.append((x, y))
    logging.info("mcts.q = {}".format(t.root.q()))
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
        policy = np.ones((CHESSBOARD_SIZE, CHESSBOARD_SIZE)) \
            / CHESSBOARD_SIZE ** 2
        value = simple_heuristics(chessboard)
        return policy, value
    t = MCTS(0, chessboard, base_policy)
    t.search(800)
    pi = t.get_pi(0)
    choices = []
    for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
        if pi[x, y] > 0:
            choices.append((x, y))
    logging.info("mcts.q = {}".format(t.root.q()))
    return choices[random.randint(0, len(choices) - 1)]


GREEDY_MCTS_PLAYER = AIPlayer(_greedy_mcts_policy)
