import itertools
import ctypes

import numpy as np

from config import CHESSBOARD_SIZE


class MCTS:
    def __init__(self, me, chessboard, policy):
        self.me = me
        self.lib = ctypes.CDLL('bazel-bin/mcts/libcapi.so')

        char_arr_chessboard = bytearray(2 * CHESSBOARD_SIZE ** 2)
        char_arr_t = ctypes.c_char * len(char_arr_chessboard)
        for who, x, y in itertools.product(range(2), range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
            idx = (who * CHESSBOARD_SIZE + x) * CHESSBOARD_SIZE + y
            char_arr_chessboard[idx] = int(chessboard[who][x][y] > 0)

        callback_t = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        )

        global callback
        @callback_t
        def callback(chessboard, out_p_ptr, out_v_ptr):
            chessboard_numpy = np.zeros((2, CHESSBOARD_SIZE, CHESSBOARD_SIZE))\
                .astype(np.float32)
            for who, x, y in itertools.product(range(2), range(CHESSBOARD_SIZE, CHESSBOARD_SIZE)):
                idx = (who * CHESSBOARD_SIZE + x) * CHESSBOARD_SIZE + y
                if int(chessboard[idx]) > 0:
                    chessboard_numpy[who][x][y] = 1

            p, v = policy(chessboard_numpy)

            p = p.reshape((-1, )).astype(ctypes.c_double)
            ctypes.memmove(
                out_p_ptr, p.ctypes.get_data(),
                ctypes.sizeof(ctypes.c_double) * len(p)
            )
            out_v_ptr[0] = ctypes.c_double(v)

        self.lib.MCTS_new.argtypes = [char_arr_t, callback_t]
        self.lib.MCTS_new.restype = ctypes.c_void_p
        self.lib.MCTS_Search.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.MCTS_Search.restype = None
        self.lib.MCTS_GetPi.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        self.lib.MCTS_GetPi.restype = None

        self.handle = self.lib.MCTS_new(
            char_arr_t.from_buffer(char_arr_chessboard),
            callback
        )

    def search(self, num_sims: int):
        self.lib.MCTS_Search(self.handle, ctypes.c_int(num_sims))

    def get_pi(self, temperature):
        pi = (ctypes.c_double * (CHESSBOARD_SIZE ** 2))()
        self.lib.MCTS_GetPi(
            self.handle, ctypes.c_double(temperature),
            ctypes.cast(pi, ctypes.POINTER(ctypes.c_double))
        )
        pi = np.array(pi).reshape((CHESSBOARD_SIZE, CHESSBOARD_SIZE))\
            .astype(np.float32)
        return pi
