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

void MCTS_GetPi(MCTS* handle, double temperature, double* out) {
  handle->GetPi(temperature, out);
}

void MCTS_delete(MCTS* handle) { delete handle; }

struct Config {
  int chessboard_size, in_a_row;
};

Config global_GetConfig() {
  return {
      .chessboard_size = CHESSBOARD_SIZE,
      .in_a_row = IN_A_ROW,
  };
}
}