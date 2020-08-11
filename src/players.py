from typing import Tuple
import threading
from copy import copy
import random
import numpy as np
import itertools

from utils import stone_is_valid
from constants import CHESSBOARD_SIZE


class Player:
    def place_stone(self, x, y):
        ...

    def evaluate(self, chessboard) -> Tuple[int, int]:
        ...


class HumanPlayer(Player):
    def __init__(self):
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.choice = None

    def place_stone(self, x, y):
        self.cv.acquire()
        self.choice = (x, y)
        self.cv.notify_all()
        self.cv.release()

    def evaluate(self, chessboard):
        self.cv.acquire()
        self.choice = None
        while self.choice is None or not stone_is_valid(chessboard, *self.choice):
            self.cv.wait()
        choice = copy(self.choice)
        self.cv.release()

        return choice


class AIPlayer(Player):
    def __init__(self, policy):
        self.policy = policy

    def evaluate(self, chessboard):
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
    from mcts import MCTS

    def base_policy(_):
        return np.ones((1, CHESSBOARD_SIZE, CHESSBOARD_SIZE)) / CHESSBOARD_SIZE ** 2, np.array([[0]])
    t = MCTS(0, chessboard, base_policy)
    t.search(100)
    pi = t.get_pi(0)
    choices = []
    for x, y in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
        if pi[x, y] > 0:
            choices.append((x, y))
    return choices[random.randint(0, len(choices)-1)]


BASIC_MCTS_PLAYER = AIPlayer(_basic_mcts_policy)
