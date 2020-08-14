#ifndef MCTS_MCTS_NODE_H_
#define MCTS_MCTS_NODE_H_

#include <stdint.h>

#include <functional>
#include <memory>
#include <vector>

#include "chessboard.h"

class MCTSNode {
 public:
  using PolicyCallback = std::function<void(const Chessboard &,
                                            std::vector<double> *p, double *v)>;

  MCTSNode(const Chessboard &chessboard, const PolicyCallback &policy);

  bool Expand(int x, int y);

  void Backup(double delta_v);

  inline double q() const { return sigma_v_ / n_; }

  std::pair<int, int> Select(double cpuct);

 private:
  inline int Index(int x, int y) { return x * chessboard_size_ + y; }

  int chessboard_size_;
  Chessboard chessboard_;
  std::vector<std::unique_ptr<MCTSNode>> childs_;
  const PolicyCallback &policy_;
  bool terminated_;
  std::vector<double> p_;
  double v_;

  double sigma_v_;
  int n_;
};

#endif