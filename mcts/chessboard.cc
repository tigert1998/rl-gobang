#include "chessboard.h"

#include <algorithm>
#include <memory>

#include "config.h"

int Chessboard::GetWinner() const {
  int tot = 0;
  for (int who : {0, 1})
    for (int x = 0; x < CHESSBOARD_SIZE; x++)
      for (int y = 0; y < CHESSBOARD_SIZE; y++) {
        tot += data_[Index(who, x, y)] > 0;
        for (int d = 0; d < 4; d++) {
          for (int i = 0; i < IN_A_ROW; i++) {
            int nx = x + DIRS[d][0] * i;
            int ny = y + DIRS[d][1] * i;
            if (std::min(nx, ny) < 0 || std::max(nx, ny) >= CHESSBOARD_SIZE) {
              goto fail_tag;
            } else if (data_[Index(who, nx, ny)] == 0) {
              goto fail_tag;
            }
          }
          return who;
        fail_tag:;
        }
      }

  if (tot >= CHESSBOARD_SIZE * CHESSBOARD_SIZE) {
    return -2;
  }

  return -1;
}

void Chessboard::SetMemory(char *ptr) {
  std::copy(ptr, ptr + 2 * CHESSBOARD_SIZE * CHESSBOARD_SIZE, data_);
}

void Chessboard::Debug() {
  for (int x = 0; x < CHESSBOARD_SIZE; x++) {
    for (int y = 0; y < CHESSBOARD_SIZE; y++) {
      char c = '.';
      if (data_[Index(0, x, y)] > 0) {
        c = 'x';
      } else if (data_[Index(1, x, y) > 0]) {
        c = 'o';
      }
      printf("%c ", c);
    }
    printf("\n");
  }
  fflush(stdout);
}