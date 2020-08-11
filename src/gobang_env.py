import threading
from typing import List
import logging
import sys

import pygame
import numpy as np

from constants import CHESSBOARD_SIZE
from atomic_value import AtomicValue
from players import Player, HUMAN_PLAYER, BASIC_MCTS_PLAYER
from utils import stone_is_valid, get_winner


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
            who = 0
            while not killed.load():
                x, y = self.players[who].evaluate(chessboard)
                if not stone_is_valid(chessboard, x, y):
                    msg = "Invalid stone placed by {} player at ({}, {})".format(
                        self.INFO[who][0], x, y)
                    logging.error(msg)
                    return
                chessboard[0, who, x, y] = 1
                self.place_stone(self.INFO[who][1], x, y)
                winner = get_winner(chessboard)
                if winner >= 0:
                    logging.info("{} player wins!".format(self.INFO[who][0]))
                    return
                elif winner == -2:
                    logging.info("The game ends in a draw.")
                    return
                who = 1 - who

        thread = threading.Thread(target=player_loop)
        thread.start()

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

    def place_stone(self, color, x, y):
        pygame.draw.circle(
            self.display,
            color,
            (20 + y * 40, 20 + x * 40), 16
        )
        pygame.display.update()


def config_log():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


if __name__ == "__main__":
    config_log()

    arena = VisualArena([BASIC_MCTS_PLAYER, HUMAN_PLAYER])
    arena.event_loop()
