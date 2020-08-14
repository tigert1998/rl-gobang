#include "mcts_node.h"

#include <cmath>

MCTSNode::MCTSNode(const Chessboard &chessboard, const PolicyCallback &policy)
    : chessboard_(chessboard), policy_(policy) {
  int winner = chessboard_.GetWinner();
  terminated_ = winner != -1;

  if (terminated_) {
    v_ = winner == -2 ? 0 : (winner == 0 ? 1 : -1);
  } else {
    policy_(chessboard_, p_, &v_);
  }

  n_ = 0;
  sigma_v_ = 0;
}

bool MCTSNode::Expand(int x, int y) {
  if (childs_[Index(x, y)] != nullptr) return false;

  Chessboard new_chessboard;
  for (int x = 0; x < CHESSBOARD_SIZE; x++)
    for (int y = 0; y < CHESSBOARD_SIZE; y++) {
      for (int who : {0, 1})
        if (chessboard_.At(who, x, y) > 0) {
          new_chessboard.Set(1 - who, x, y);
          break;
        }
    }
  new_chessboard.Set(1, x, y);

  childs_[Index(x, y)].reset(new MCTSNode(new_chessboard, policy_));
  return true;
}

void MCTSNode::Backup(double delta_v) {
  n_ += 1;
  sigma_v_ += delta_v;
}

std::pair<int, int> MCTSNode::Select(double cpuct) {
  std::pair<int, int> ans;
  double highest = -1e10;

  for (int x = 0; x < CHESSBOARD_SIZE; x++)
    for (int y = 0; y < CHESSBOARD_SIZE; y++) {
      if (chessboard_.At(0, x, y) + chessboard_.At(1, x, y) > 0) {
        continue;
      }
      double p = p_[Index(x, y)];
      double tmp = cpuct * p * std::pow(n_, 0.5);
      auto child = childs_[Index(x, y)].get();
      if (child != nullptr && child->n_ > 0) {
        tmp = -child->q() + tmp / (child->n_ + 1);
      }

      if (tmp > highest) {
        highest = tmp;
        ans = {x, y};
      }
    }

  return ans;
}