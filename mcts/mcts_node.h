#ifndef MCTS_MCTS_NODE_H_
#define MCTS_MCTS_NODE_H_

#include <stdint.h>

#include <functional>
#include <memory>
#include <vector>

#include "chessboard.h"
#include "config.h"
#include "static_queue.h"

class MCTSNode {
  friend class MCTS;

 public:
  MCTSNode(const Chessboard &chessboard, MCTSNode *father);

  bool Expand(int x, int y);

  void Backup(double delta_v);

  inline double q() const { return sigma_v_ / std::max(n_, 1); }

  inline int n() const { return n_; }

  std::pair<int, int> Select(double cpuct, double vloss);

  inline bool terminated() const { return terminated_; }

  inline MCTSNode *child(int x, int y) { return childs_[Index(x, y)].get(); }

  inline std::unique_ptr<MCTSNode> child_ownership(int x, int y) {
    return std::move(childs_[Index(x, y)]);
  }

  inline double v() const { return v_; }

  inline MCTSNode *father() const { return father_; }
  inline void set_father(MCTSNode *father) { father_ = father; }

  inline bool evaluating() const { return evaluating_; }

  inline void set_p_noise(double *p_noise) { p_noise_ = p_noise; }

  inline void inc_vloss_cnt() { vloss_cnt_ += 1; }
  inline void dec_vloss_cnt() { vloss_cnt_ -= 1; }

  Chessboard chessboard() const { return chessboard_; }

 private:
  inline int Index(int x, int y) { return x * CHESSBOARD_SIZE + y; }

  Chessboard chessboard_;
  MCTSNode *father_;
  std::unique_ptr<MCTSNode> childs_[CHESSBOARD_SIZE * CHESSBOARD_SIZE];

  bool terminated_;
  bool evaluating_;

  double *p_noise_ = nullptr;
  double p_[CHESSBOARD_SIZE * CHESSBOARD_SIZE];
  double v_;

  double sigma_v_;
  int n_;
  int vloss_cnt_;
};

#endif