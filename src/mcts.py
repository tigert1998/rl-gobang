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
            ctypes.POINTER(ctypes.c_byte),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        )

        global callback
        @callback_t
        def callback(chessboard, out_p_ptr, out_v_ptr):
            p, v = policy(self._byte_ptr_to_chessboard(chessboard))

            p = p.reshape((-1, )).astype(ctypes.c_double)
            ctypes.memmove(
                out_p_ptr, p.ctypes.get_data(),
                ctypes.sizeof(ctypes.c_double) * len(p)
            )
            out_v_ptr[0] = ctypes.c_double(v)

        self.lib.MCTS_new.argtypes = [char_arr_t, callback_t]
        self.lib.MCTS_new.restype = ctypes.c_void_p
        self.lib.MCTS_Search.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        self.lib.MCTS_Search.restype = None
        self.lib.MCTS_GetPi.argtypes = \
            [ctypes.c_void_p, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        self.lib.MCTS_GetPi.restype = None
        self.lib.MCTS_terminated.argtypes = [ctypes.c_void_p]
        self.lib.MCTS_terminated.restype = ctypes.c_bool
        self.lib.MCTS_chessboard.argtypes = \
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_byte)]
        self.lib.MCTS_chessboard.restype = None
        self.lib.MCTS_StepForward.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.MCTS_StepForward.restype = None
        self.lib.MCTS_v.argtypes = [ctypes.c_void_p]
        self.lib.MCTS_v.restype = ctypes.c_double
        self.lib.MCTS_delete.argtypes = [ctypes.c_void_p]
        self.lib.MCTS_delete.restype = None

        self.handle = self.lib.MCTS_new(
            char_arr_t.from_buffer(char_arr_chessboard),
            callback
        )

    def search(self, num_sims: int, cpuct: float):
        self.lib.MCTS_Search(
            self.handle, ctypes.c_int(num_sims),
            ctypes.c_double(cpuct)
        )

    def get_pi(self, temperature):
        pi = (ctypes.c_double * (CHESSBOARD_SIZE ** 2))()
        self.lib.MCTS_GetPi(
            self.handle, ctypes.c_double(temperature),
            ctypes.cast(pi, ctypes.POINTER(ctypes.c_double))
        )
        pi = np.array(pi).reshape((CHESSBOARD_SIZE, CHESSBOARD_SIZE))\
            .astype(np.float32)
        return pi

    def step_forward(self, x, y):
        self.lib.MCTS_StepForward(
            self.handle, ctypes.c_int(x), ctypes.c_int(y))

    def terminated(self) -> bool:
        return bool(self.lib.MCTS_terminated(self.handle))

    def chessboard(self) -> np.array:
        byte_arr = (ctypes.c_byte * (2 * CHESSBOARD_SIZE ** 2))()
        byte_ptr = ctypes.cast(byte_arr, ctypes.POINTER(ctypes.c_byte))
        self.lib.MCTS_chessboard(self.handle, byte_ptr)
        return self._byte_ptr_to_chessboard(byte_ptr)

    def v(self) -> np.float32:
        return np.float32(self.lib.MCTS_v(self.handle))

    def __del__(self):
        self.lib.MCTS_delete(self.handle)

    @classmethod
    def _byte_ptr_to_chessboard(cls, ptr) -> np.array:
        ret = np.zeros((2, CHESSBOARD_SIZE, CHESSBOARD_SIZE))\
                .astype(np.float32)
        for who, x, y in itertools.product(
                range(2), range(CHESSBOARD_SIZE), range(CHESSBOARD_SIZE)):
            idx = (who * CHESSBOARD_SIZE + x) * CHESSBOARD_SIZE + y
            if int(ptr[idx]) > 0:
                ret[who][x][y] = 1
        return ret
