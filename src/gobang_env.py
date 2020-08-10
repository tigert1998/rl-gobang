import pygame

import numpy as np
import threading
import itertools
from typing import List, Tuple
from copy import copy
import logging
import random

from constants import CHESSBOARD_SIZE
from atomic_value import AtomicValue


def stone_is_valid(chessboard, x, y) -> bool:
    if min(x, y) < 0 or max(x, y) >= CHESSBOARD_SIZE:
        return False
    if chessboard[0, ::, x, y].sum() > 0:
        return False
    return True


def get_winner(chessboard):
    assert chessboard.shape == (1, 2, CHESSBOARD_SIZE, CHESSBOARD_SIZE)

    dirs = [[0, 1], [-1, 1], [-1, 0], [-1, -1]]

    for a in [0, 1]:
        for x, y, d in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE), dirs):
            yes = True
            for i in range(5):
                nx, ny = np.array([x, y]) + np.array(d) * i
                if min(nx, ny) < 0 or max(nx, ny) >= CHESSBOARD_SIZE:
                    yes = False
                else:
                    yes &= chessboard[0, a, nx, ny] > 0
            if yes:
                return a

    return None


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


class VisualArena:
    INFO = {
        0: ("black", (0, 0, 0)),
        1: ("white", (255, 255, 255))
    }

    def __init__(self, players: List[Player]):
        self.players = players
        self.display = pygame.display.set_mode((600, 600))
        pygame.display.set_caption('Gobang')
        background_img = pygame.image.load('imgs/chessboard.png')
        self.display.blit(background_img, (0, 0))
        pygame.display.update()

    def event_loop(self):
        killed = AtomicValue(False)

        def player_loop():
            chessboard = np.zeros((1, 2, CHESSBOARD_SIZE, CHESSBOARD_SIZE))
            round = 0
            while not killed.load():
                x, y = self.players[round].evaluate(chessboard)
                if not stone_is_valid(chessboard, x, y):
                    logging.fatal(
                        f"invalid stone placed by {self.INFO[round][0]} player at ({x}, {y})"
                    )
                    return
                chessboard[0, round, x, y] = 1
                self.place_stone(self.INFO[round][1], x, y)
                if get_winner(chessboard) is not None:
                    logging.info(f"{self.INFO[round][0]} player wins!")
                    return
                round = 1 - round

        thread = threading.Thread(target=player_loop)
        thread.start()

        # pylint: disable=no-member
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                killed.store(True)
                pygame.quit()
                thread.join()
                break
            elif event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                x, y = np.round((np.array(pos) - 20) / 40)[::-1]\
                    .astype(np.int32)
                for i in range(2):
                    self.players[i].place_stone(x, y)
        # pylint: enable=no-member

    def place_stone(self, color, x, y):
        pygame.draw.circle(
            self.display,
            color,
            (20 + y * 40, 20 + x * 40), 16
        )
        pygame.display.update()


if __name__ == "__main__":
    import sys
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    human_player = HumanPlayer()

    def policy(chessboard):
        while True:
            x = random.randint(0, CHESSBOARD_SIZE-1)
            y = random.randint(0, CHESSBOARD_SIZE-1)
            if stone_is_valid(chessboard, x, y):
                return (x, y)
    ai_player = AIPlayer(policy)

    arena = VisualArena([ai_player, ai_player])
    arena.event_loop()
