#include "mcts.h"

#include <cmath>
#include <iostream>

MCTS::MCTS(const Chessboard& chessboard, const MCTSNode::PolicyCallback& policy)
    : chessboard_(chessboard), policy_(policy), root_(nullptr) {}

void MCTS::Search(int num_sims) {
  if (root_ == nullptr) {
    root_.reset(new MCTSNode(chessboard_, policy_));
  }

  for (int i = 0; i < num_sims; i++) {
    Simulate();
  }
}

void MCTS::Simulate() {
  auto node = root_.get();

  double cpuct = 3;

  bool expanded = false;

  MCTSNode* path[CHESSBOARD_SIZE * CHESSBOARD_SIZE];
  path[0] = node;
  int path_size = 1;

  for (; !node->terminated() && !expanded;) {
    auto xy = node->Select(cpuct);
    expanded = node->Expand(xy.first, xy.second);
    node = node->child(xy.first, xy.second);
    path[path_size++] = node;
  }

  double delta_v = node->v();
  for (int i = path_size - 1; i >= 0; i--) {
    node->Backup(delta_v);
    delta_v = -delta_v;
  }
}

void MCTS::StepForward(int x, int y) {
  chessboard_ = root_->child(x, y)->chessboard();
  root_ = root_->child_ownership(x, y);
}

void MCTS::GetPi(double temperature, double* out) {
  for (int i = 0; i < CHESSBOARD_SIZE * CHESSBOARD_SIZE; i++) out[i] = 0;
  double deno = 0;
  int highest = 0;
  std::vector<std::pair<int, int>> pos;

  double eps = 1e-6;

  for (int x = 0; x < CHESSBOARD_SIZE; x++)
    for (int y = 0; y < CHESSBOARD_SIZE; y++) {
      auto child = root_->child(x, y);
      if (child == nullptr) {
        continue;
      }
      if (temperature < eps) {
        if (child->n() > highest) {
          pos.clear();
          pos.push_back({x, y});
        } else if (child->n() == highest) {
          pos.push_back({x, y});
        }
      } else {
        int idx = x * CHESSBOARD_SIZE + y;
        out[idx] = pow(child->n(), 1 / temperature);
        deno += out[idx];
      }
    }

  if (temperature < eps) {
    for (auto xy : pos) {
      int idx = xy.first * CHESSBOARD_SIZE + xy.second;
      out[idx] = 1.0 / pos.size();
    }
  } else {
    for (int i = 0; i < CHESSBOARD_SIZE * CHESSBOARD_SIZE; i++) out[i] /= deno;
  }
}