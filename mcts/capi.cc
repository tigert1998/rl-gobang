#include "config.h"
#include "mcts.h"

extern "C" {

MCTS* MCTS_new(char* chessboard, double vloss, int batch_size,
               void (*callback)(int, char**, double**, double**)) {
  Chessboard new_chessboard;
  new_chessboard.SetMemory(chessboard);
  return new MCTS(new_chessboard, vloss, batch_size, callback);
}

void MCTS_Search(MCTS* handle, int num_sims, double cpuct,
                 double dirichlet_alpha) {
  handle->Search(num_sims, cpuct, dirichlet_alpha);
}

void MCTS_StepForward(MCTS* handle, int x, int y) { handle->StepForward(x, y); }

void MCTS_GetPi(MCTS* handle, double temperature, double* out) {
  handle->GetPi(temperature, out);
}

bool MCTS_terminated(MCTS* handle) { return handle->terminated(); }

void MCTS_chessboard(MCTS* handle, char* ptr) { handle->chessboard(ptr); }

void MCTS_delete(MCTS* handle) { delete handle; }

double MCTS_v(MCTS* handle) { return handle->v(); }

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