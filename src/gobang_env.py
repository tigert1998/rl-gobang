import threading
from typing import List
import logging
import itertools

import pygame
import numpy as np

from constants import CHESSBOARD_SIZE
from atomic_value import AtomicValue
from players import *
from utils import stone_is_valid, get_winner, config_log


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

    def event_loop(self, initial_chessboard=None, initial_player=None):
        killed = AtomicValue(False)

        chessboard = initial_chessboard
        if chessboard is None:
            chessboard = np.zeros((2, CHESSBOARD_SIZE, CHESSBOARD_SIZE))
        chessboard = chessboard.astype(np.float32)
        for x, y, who in itertools.product(range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE), range(2)):
            if chessboard[who, x, y] > 0:
                self.place_stone(who, x, y)
        if initial_player is None:
            who = 1 if (chessboard[0, :, :].sum() >
                        chessboard[1, :, :].sum()) else 0
        else:
            who = initial_player

        def player_loop(chessboard, who):
            while not killed.load():
                choice = self.players[who].evaluate(who, chessboard)
                if killed.load():
                    break
                x, y = choice
                if not stone_is_valid(chessboard, x, y):
                    msg = "Invalid stone placed by {} player at ({}, {})".format(
                        self.INFO[who][0], x, y)
                    logging.error(msg)
                    break
                chessboard[who, x, y] = 1
                self.place_stone(who, x, y)
                winner = get_winner(chessboard)
                if winner >= 0:
                    logging.info("{} player wins!".format(self.INFO[who][0]))
                    break
                elif winner == -2:
                    logging.info("The game ends in a draw.")
                    break
                who = 1 - who
            logging.info("chessboard = {}".format(chessboard))

        thread = threading.Thread(target=player_loop, args=(chessboard, who))
        thread.start()

        pygame.display.update()
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                killed.store(True)
                for i in range(2):
                    self.players[i].kill()
                pygame.quit()
                thread.join()
                break
            elif event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                x, y = np.round((np.array(pos) - 20) / 40)[::-1]\
                    .astype(np.int32)
                for i in range(2):
                    self.players[i].place_stone(x, y)

    def place_stone(self, who, x, y):
        pygame.draw.circle(
            self.display,
            self.INFO[who][1],
            (20 + y * 40, 20 + x * 40), 16
        )
        pygame.display.update()


if __name__ == "__main__":
    config_log()

    player = NNMCTSAIPlayer("/Users/tigertang/Desktop/250.pt")
    arena = VisualArena([HUMAN_PLAYER, player])
    arena.event_loop()
