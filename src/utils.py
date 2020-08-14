import sys
import itertools
import logging

import numpy as np

from config import CHESSBOARD_SIZE, IN_A_ROW

_DIRS = [[0, 1], [-1, 1], [-1, 0], [-1, -1]]


def stone_is_valid(chessboard, x, y) -> bool:
    if min(x, y) < 0 or max(x, y) >= CHESSBOARD_SIZE:
        return False
    if chessboard[:, x, y].sum() > 0:
        return False
    return True


def get_winner(chessboard):
    """Get winner of the chessboard.

    Args:
        chessboard: A np.array of shape (1, 2, CHESSBOARD_SIZE, CHESSBOARD_SIZE).

    Returns:
        The winner of the state.
        0 or 1 represent the winner.
        -1 represents the game should continue.
        Besides, -2 means the game has ended but there is no winner.
    """
    assert chessboard.shape == (2, CHESSBOARD_SIZE, CHESSBOARD_SIZE)

    for a in [0, 1]:
        for x, y, d in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE), _DIRS):
            yes = True
            for i in range(IN_A_ROW):
                nx, ny = np.array([x, y]) + np.array(d) * i
                if min(nx, ny) < 0 or max(nx, ny) >= CHESSBOARD_SIZE:
                    yes = False
                else:
                    yes &= chessboard[a, nx, ny] > 0
            if yes:
                return a

    if chessboard.sum() >= CHESSBOARD_SIZE ** 2:
        return -2

    return -1


def simple_heuristics(chessboard) -> float:
    assert chessboard.shape == (2, CHESSBOARD_SIZE, CHESSBOARD_SIZE)

    heuristics = [0.0, 0.0]
    rank = [0, 0, 1, 1e2, 1e4, 1e6]

    for who in [0, 1]:
        for x, y, d in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE), _DIRS):
            break_flag = False
            for i in range(IN_A_ROW):
                nx, ny = np.array([x, y]) + np.array(d) * i
                if min(nx, ny) < 0 or max(nx, ny) >= CHESSBOARD_SIZE or not (chessboard[who, nx, ny] > 0):
                    break_flag = True
                    break
            if not break_flag:
                i += 1
            heuristics[who] += rank[i]

    deno = heuristics[0] + heuristics[1]
    if not (deno > 0):
        return 0
    return 2 * heuristics[0] / deno - 1


def config_log():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
