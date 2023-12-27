#include "config.h"
#include "mcts.h"

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

extern "C" {

API MCTS* MCTS_new(char* chessboard, double vloss, int batch_size,
                   void (*callback)(int, char**, double**, double**)) {
  Chessboard new_chessboard;
  new_chessboard.SetMemory(chessboard);
  return new MCTS(new_chessboard, vloss, batch_size, callback);
}

API void MCTS_Search(MCTS* handle, int num_sims, double cpuct,
                     double dirichlet_alpha) {
  handle->Search(num_sims, cpuct, dirichlet_alpha);
}

API void MCTS_StepForward(MCTS* handle, int x, int y) {
  handle->StepForward(x, y);
}

API void MCTS_GetPi(MCTS* handle, double temperature, double* out) {
  handle->GetPi(temperature, out);
}

API bool MCTS_terminated(MCTS* handle) { return handle->terminated(); }

API void MCTS_chessboard(MCTS* handle, char* ptr) { handle->chessboard(ptr); }

API void MCTS_delete(MCTS* handle) { delete handle; }

API double MCTS_v(MCTS* handle) { return handle->v(); }

struct Config {
  int chessboard_size, in_a_row;
};

API Config global_GetConfig() {
  Config config;
  config.chessboard_size = CHESSBOARD_SIZE;
  config.in_a_row = IN_A_ROW;
  return config;
}
}