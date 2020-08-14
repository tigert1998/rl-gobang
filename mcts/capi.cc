#include "config.h"
#include "mcts.h"

extern "C" {

MCTS* MCTS_new(char* chessboard,
               void (*callback)(const char*, double*, double*)) {
  Chessboard new_chessboard;
  new_chessboard.SetMemory(chessboard);
  return new MCTS(new_chessboard,
                  [callback](const Chessboard& chessboard, double* p,
                             double* v) { callback(chessboard.Data(), p, v); });
}

void MCTS_Search(MCTS* handle, int num_sims) { handle->Search(num_sims); }

void MCTS_StepForward(MCTS* handle, int x, int y) { handle->StepForward(x, y); }

void MCTS_delete(MCTS* handle) { delete handle; }
}