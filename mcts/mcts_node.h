#ifndef MCTS_MCTS_NODE_H_
#define MCTS_MCTS_NODE_H_

#include <stdint.h>

#include <functional>
#include <memory>
#include <vector>

#include "chessboard.h"
#include "config.h"

class MCTSNode {
 public:
  using PolicyCallback =
      std::function<void(Chessboard &, double *p, double *v)>;

  MCTSNode(const Chessboard &chessboard, const PolicyCallback &policy);

  bool Expand(int x, int y);

  void Backup(double delta_v);

  inline double q() const { return sigma_v_ / n_; }

  inline int n() const { return n_; }

  std::pair<int, int> Select(double cpuct);

  inline bool terminated() const { return terminated_; }

  inline MCTSNode *child(int x, int y) { return childs_[Index(x, y)].get(); }

  inline std::unique_ptr<MCTSNode> child_ownership(int x, int y) {
    return std::move(childs_[Index(x, y)]);
  }

  inline double v() const { return v_; }

  Chessboard chessboard() const { return chessboard_; }

 private:
  inline int Index(int x, int y) { return x * CHESSBOARD_SIZE + y; }

  Chessboard chessboard_;
  std::unique_ptr<MCTSNode> childs_[CHESSBOARD_SIZE * CHESSBOARD_SIZE];
  PolicyCallback policy_;
  bool terminated_;
  double p_[CHESSBOARD_SIZE * CHESSBOARD_SIZE];
  double v_;

  double sigma_v_;
  int n_;
};

#endif