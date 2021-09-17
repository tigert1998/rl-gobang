from typing import Tuple, Optional
import os
import logging
import time
import random

import numpy as np
from PIL import Image

from players import NNMCTSAIPlayer
from gobang_utils import config_log
from gobang_vis import chessboard_str

from config import ADB

def get_median_color(img, xy, radius) -> np.array:
    img = img.crop(
        (xy[0] - radius, xy[1] - radius,
         xy[0] + radius, xy[1] + radius)
    )
    arr = np.array(img.getdata())
    color = np.median(arr, axis=0)[:-1]
    return color


class OnlinePlatform:
    def __init__(self, adb_device_id: Optional[str]):
        self.adb_device_id = adb_device_id

    def _device_str(self):
        if self.adb_device_id is None:
            return ""
        else:
            return "-s {}".format(self.adb_device_id)

    def _shell(self, cmd):
        cmd = "{} {} shell {}".format(ADB, self._device_str(), cmd)
        ret = os.system(cmd)
        if ret != 0:
            logging.warning("{} = {}".format(ret, cmd))
        return ret

    def _pull(self, path):
        try:
            os.makedirs("tmp")
        except FileExistsError:
            ...
        cmd = "{} {} pull {} tmp".format(ADB, self._device_str(), path)
        ret = os.system(cmd)
        if ret != 0:
            logging.warning("{} = {}".format(ret, cmd))
        return ret

    def wait_on_chessboard(self) -> Tuple[int, np.array]:
        while True:
            ret = 1
            while ret != 0:
                ret = self._shell("screencap -p /sdcard/tmp/screenshot.png")
                time.sleep(0.1)
            ret = 1
            while ret != 0:
                ret = self._pull("/sdcard/tmp/screenshot.png")
                time.sleep(0.1)

            who, chessboard = self._detect_chessboard("tmp/screenshot.png")
            if who == 0:
                if chessboard[1, :, :].sum() >= chessboard[0, :, :].sum():
                    return who, chessboard
            elif who == 1:
                if chessboard[0, :, :].sum() > chessboard[1, :, :].sum():
                    return who, chessboard
            else:
                assert False

            logging.info("waiting for the opponent to place stone")
            time.sleep(0.5)

    def _detect_chessboard(self, img_path: str) -> Tuple[int, np.array]:
        raise NotImplementedError()
        return -1, np.empty((0,))

    def place_stone(self, x, y):
        raise NotImplementedError()


class TencentHappyGomoku(OnlinePlatform):
    """TencentHappyGomoku
    The class is used to automatically play at "欢乐五子棋腾讯版小程序".
    Now it is only tested under MI 8 SE.
    """

    def _chess_coordiante_at(self, x, y):
        left = 42
        top = 696
        grid_height = 930.0 / 14
        margin = 33
        return (y * grid_height + margin + left, x * grid_height + margin + top)

    def place_stone(self, x, y):
        screen_x, screen_y = self._chess_coordiante_at(x, y)
        cmd = "input tap {} {}".format(screen_x, screen_y)
        self._shell(cmd)
        time.sleep(0.1 + random.random() * 0.1)
        self._shell(cmd)

    def _detect_chessboard(self, img_path: str) -> Tuple[int, np.array]:
        img = Image.open(img_path)
        chess_radius = 57.0 / 2

        chessboard = np.zeros((2, 15, 15)).astype(np.float32)

        for i in range(15):
            for j in range(15):
                c = get_median_color(
                    img, self._chess_coordiante_at(i, j),
                    chess_radius
                )
                avg_c = np.average(c) / 255

                if avg_c < 0.65:
                    # black
                    chessboard[0, i, j] = 1
                elif avg_c > 0.9:
                    # white
                    chessboard[1, i, j] = 1

        ROLE_LOCS = [(858, 1860), (230, 418)]
        roles = [-1] * 2
        for i in range(2):
            c = get_median_color(img, ROLE_LOCS[i], chess_radius)
            avg_c = np.average(c) / 255
            if avg_c < 0.65:
                # black
                roles[i] = 0
            elif avg_c > 0.9:
                roles[i] = 1
            else:
                assert False
        assert roles[0] + roles[1] == 1

        return roles[0], chessboard


if __name__ == "__main__":
    config_log(None)
    platform = TencentHappyGomoku(None)
    player = NNMCTSAIPlayer("/home/fucong/playground/rl-gobang/41270.pt")

    while True:
        who, chessboard = platform.wait_on_chessboard()
        logging.info("\n" + chessboard_str(chessboard))
        x, y = player.evaluate(who, chessboard)
        logging.info("the agent is placing stone at ({}, {})".format(x, y))
        platform.place_stone(x, y)
