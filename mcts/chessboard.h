#ifndef MCTS_CHESSBOARD_H_
#define MCTS_CHESSBOARD_H_

#include <vector>

class Chessboard {
 public:
  inline Chessboard(int chessboard_size) : chessboard_size_(chessboard_size) {
    data_.resize(2 * chessboard_size * chessboard_size);
  }

  inline void Set(int c, int x, int y) { data_[Index(c, x, y)] = 1; }

  inline int At(int c, int x, int y) const {
    return (int)data_[Index(c, x, y)];
  }

  inline int Size() const { return chessboard_size_; }

  int GetWinner() const;

 private:
  inline int Index(int c, int x, int y) const {
    return (c * chessboard_size_ + x) * chessboard_size_ + y;
  }

  std::vector<char> data_;
  int chessboard_size_;
};

#endif