from typing import Tuple
import threading
from copy import copy
import random
import numpy as np
import itertools

from utils import stone_is_valid
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
            chessboard = chessboard[:, ::-1, :, :]
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
        policy = np.ones((1, CHESSBOARD_SIZE, CHESSBOARD_SIZE)) \
            / CHESSBOARD_SIZE ** 2
        value = np.array([[0]])
        return policy, value
    t = MCTS(0, chessboard, base_policy)
    t.search(1600)
    pi = t.get_pi(0)
    choices = []
    for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
        if pi[x, y] > 0:
            choices.append((x, y))
    return choices[random.randint(0, len(choices) - 1)]


BASIC_MCTS_PLAYER = AIPlayer(_basic_mcts_policy)
