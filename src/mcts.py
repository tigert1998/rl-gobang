import sys
import itertools
from ctypes import *
from typing import Optional

import numpy as np

from config import CHESSBOARD_SIZE


class MCTS:
    def __init__(self, chessboard, vloss, batch_size, policy):
        self.lib = CDLL(
            "bazel-bin/mcts/capi_shared.dll"
            if sys.platform.startswith("win")
            else "bazel-bin/mcts/capi_shared.so"
        )

        char_arr_chessboard = bytearray(2 * CHESSBOARD_SIZE**2)
        char_arr_t = c_char * len(char_arr_chessboard)
        for who, x, y in itertools.product(
            range(2), range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)
        ):
            idx = (who * CHESSBOARD_SIZE + x) * CHESSBOARD_SIZE + y
            char_arr_chessboard[idx] = int(chessboard[who][x][y] > 0)

        callback_t = CFUNCTYPE(
            None,
            c_int,
            POINTER(POINTER(c_byte)),
            POINTER(POINTER(c_double)),
            POINTER(POINTER(c_double)),
        )

        global callback

        @callback_t
        def callback(n, chessboards, probs, vs):
            i = np.stack(
                [self._byte_ptr_to_chessboard(chessboards[i]) for i in range(n)]
            )
            x, y = policy(i)
            for i in range(n):
                memmove(
                    probs[i],
                    x[i].reshape((-1,)).astype(c_double).ctypes.get_data(),
                    sizeof(c_double) * (CHESSBOARD_SIZE**2),
                )
                vs[i][0] = c_double(y[i])

        self.lib.MCTS_new.argtypes = [char_arr_t, c_double, c_int, callback_t]
        self.lib.MCTS_new.restype = c_void_p
        self.lib.MCTS_Search.argtypes = [c_void_p, c_int, c_double, c_double]
        self.lib.MCTS_Search.restype = None
        self.lib.MCTS_GetPi.argtypes = [c_void_p, c_double, POINTER(c_double)]
        self.lib.MCTS_GetPi.restype = None
        self.lib.MCTS_terminated.argtypes = [c_void_p]
        self.lib.MCTS_terminated.restype = c_bool
        self.lib.MCTS_chessboard.argtypes = [c_void_p, POINTER(c_byte)]
        self.lib.MCTS_chessboard.restype = None
        self.lib.MCTS_StepForward.argtypes = [c_void_p, c_int, c_int]
        self.lib.MCTS_StepForward.restype = None
        self.lib.MCTS_v.argtypes = [c_void_p]
        self.lib.MCTS_v.restype = c_double
        self.lib.MCTS_delete.argtypes = [c_void_p]
        self.lib.MCTS_delete.restype = None

        self.handle = self.lib.MCTS_new(
            char_arr_t.from_buffer(char_arr_chessboard),
            c_double(vloss),
            c_int(batch_size),
            callback,
        )

    def search(self, num_sims: int, cpuct: float, alpha: Optional[float]):
        if alpha is None:
            alpha = -1
        self.lib.MCTS_Search(
            self.handle, c_int(num_sims), c_double(cpuct), c_double(alpha)
        )

    def get_pi(self, temperature):
        pi = (c_double * (CHESSBOARD_SIZE**2))()
        self.lib.MCTS_GetPi(
            self.handle, c_double(temperature), cast(pi, POINTER(c_double))
        )
        pi = np.array(pi).reshape((CHESSBOARD_SIZE, CHESSBOARD_SIZE)).astype(np.float32)
        return pi

    def step_forward(self, x, y):
        self.lib.MCTS_StepForward(self.handle, c_int(x), c_int(y))

    def terminated(self) -> bool:
        return bool(self.lib.MCTS_terminated(self.handle))

    def chessboard(self) -> np.array:
        byte_arr = (c_byte * (2 * CHESSBOARD_SIZE**2))()
        byte_ptr = cast(byte_arr, POINTER(c_byte))
        self.lib.MCTS_chessboard(self.handle, byte_ptr)
        return self._byte_ptr_to_chessboard(byte_ptr)

    def v(self) -> np.float32:
        return np.float32(self.lib.MCTS_v(self.handle))

    def __del__(self):
        self.lib.MCTS_delete(self.handle)

    @classmethod
    def _byte_ptr_to_chessboard(cls, ptr) -> np.array:
        ret = np.zeros((2, CHESSBOARD_SIZE, CHESSBOARD_SIZE)).astype(np.float32)
        for who, x, y in itertools.product(
            range(2), range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)
        ):
            idx = (who * CHESSBOARD_SIZE + x) * CHESSBOARD_SIZE + y
            if int(ptr[idx]) > 0:
                ret[who][x][y] = 1
        return ret
