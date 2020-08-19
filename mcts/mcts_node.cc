#include "mcts_node.h"

#include <cmath>

MCTSNode::MCTSNode(const Chessboard &chessboard, MCTSNode *father)
    : chessboard_(chessboard), father_(father) {
  std::fill(childs_, childs_ + CHESSBOARD_SIZE * CHESSBOARD_SIZE, nullptr);

  int winner = chessboard_.GetWinner();
  terminated_ = winner != -1;
  if (terminated_) {
    v_ = winner == -2 ? 0 : (winner == 0 ? 1 : -1);
    evaluating_ = false;
  } else {
    evaluating_ = true;
  }

  n_ = 0;
  sigma_v_ = 0;
  vloss_cnt_ = 0;
}

bool MCTSNode::Expand(int x, int y) {
  if (childs_[Index(x, y)] != nullptr) return false;

  int half = CHESSBOARD_SIZE * CHESSBOARD_SIZE;
  Chessboard new_chessboard;
  std::copy(chessboard_.Data(), chessboard_.Data() + half,
            new_chessboard.Data() + half);
  std::copy(chessboard_.Data() + half, chessboard_.Data() + 2 * half,
            new_chessboard.Data());
  new_chessboard.Set(1, x, y);

  childs_[Index(x, y)].reset(new MCTSNode(new_chessboard, this));
  return true;
}

void MCTSNode::Backup(double delta_v) {
  n_ += 1;
  sigma_v_ += delta_v;
  evaluating_ = false;
}

std::pair<int, int> MCTSNode::Select(double cpuct, double vloss) {
  std::pair<int, int> ans;
  double highest = -1e10;

  bool use_p_noise = p_noise_ != nullptr;

  for (int x = 0; x < CHESSBOARD_SIZE; x++)
    for (int y = 0; y < CHESSBOARD_SIZE; y++) {
      if (chessboard_.At(0, x, y) + chessboard_.At(1, x, y) > 0) {
        continue;
      }
      int idx = Index(x, y);
      double p;
      if (use_p_noise) {
        const double e = 0.25;
        p = (1 - e) * p_[idx] + e * p_noise_[idx];
      } else {
        p = p_[idx];
      }

      double tmp = cpuct * p * std::pow(n_, 0.5);
      auto child = childs_[idx].get();
      if (child != nullptr) {
        tmp = -vloss * child->vloss_cnt_ / std::max(child->n_, 1) - child->q() +
              tmp / (child->n_ + 1);
      }

      if (tmp > highest) {
        highest = tmp;
        ans = {x, y};
      }
    }

  return ans;
}