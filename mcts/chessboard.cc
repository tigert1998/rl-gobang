#include "chessboard.h"

#include "config.h"

int Chessboard::GetWinner() const {
  int tot = 0;
  for (int who : {0, 1})
    for (int x = 0; x < chessboard_size_; x++)
      for (int y = 0; y < chessboard_size_; y++) {
        tot += data_[Index(who, x, y)] > 0;
        for (int d = 0; d < 4; d++) {
          bool yes = true;
          for (int i = 0; i < IN_A_ROW; i++) {
            int nx = x + DIRS[d][0] * i;
            int ny = y + DIRS[d][1] * i;
            if (std::min(nx, ny) < 0 || std::max(nx, ny) >= chessboard_size_) {
              yes = false;
            } else {
              yes &= data_.at(Index(who, nx, ny)) > 0;
            }
          }
          if (yes) {
            return who;
          }
        }
      }

  if (tot >= chessboard_size_ * chessboard_size_) {
    return -2;
  }

  return -1;
}