import itertools

import numpy as np

from constants import CHESSBOARD_SIZE, IN_A_ROW


def stone_is_valid(chessboard, x, y) -> bool:
    if min(x, y) < 0 or max(x, y) >= CHESSBOARD_SIZE:
        return False
    if chessboard[0, ::, x, y].sum() > 0:
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
    assert chessboard.shape == (1, 2, CHESSBOARD_SIZE, CHESSBOARD_SIZE)

    dirs = [[0, 1], [-1, 1], [-1, 0], [-1, -1]]

    for a in [0, 1]:
        for x, y, d in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE), dirs):
            yes = True
            for i in range(IN_A_ROW):
                nx, ny = np.array([x, y]) + np.array(d) * i
                if min(nx, ny) < 0 or max(nx, ny) >= CHESSBOARD_SIZE:
                    yes = False
                else:
                    yes &= chessboard[0, a, nx, ny] > 0
            if yes:
                return a

    if chessboard.sum() >= CHESSBOARD_SIZE ** 2:
        return -2

    return -1
