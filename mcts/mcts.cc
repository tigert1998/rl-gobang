#include "mcts.h"

#include <cmath>
#include <iostream>
#include <random>
#include <ctime>

void MCTS::EnsureRoot() {
  if (root_ == nullptr) {
    root_.reset(new MCTSNode(chessboard_, policy_));
  }
}

MCTS::MCTS(const Chessboard& chessboard, const MCTSNode::PolicyCallback& policy)
    : chessboard_(chessboard), policy_(policy), root_(nullptr) {}

void MCTS::Search(int num_sims, double cpuct, double dirichlet_alpha) {
  EnsureRoot();

  if (dirichlet_alpha > 0) {
    AllocateNoise(dirichlet_alpha);
    root_->set_p_noise(p_noise_);
  }

  for (int i = 0; i < num_sims; i++) {
    Simulate(cpuct);
  }
}

void MCTS::Simulate(double cpuct) {
  auto node = root_.get();

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
    path[i]->Backup(delta_v);
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
          highest = child->n();
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

bool MCTS::terminated() {
  EnsureRoot();
  return root_->terminated();
}

void MCTS::chessboard(char* ptr) {
  std::copy(chessboard_.Data(),
            chessboard_.Data() + 2 * CHESSBOARD_SIZE * CHESSBOARD_SIZE, ptr);
}

double MCTS::v() {
  EnsureRoot();
  return root_->v();
}

void MCTS::AllocateNoise(double alpha) {
  std::gamma_distribution<double> gamma(alpha, 1);
  std::default_random_engine rng(std::time(nullptr));

  double deno = 0;

  for (int i = 0; i < CHESSBOARD_SIZE * CHESSBOARD_SIZE; i++) {
    p_noise_[i] = gamma(rng);
    deno += p_noise_[i];
  }
  for (int i = 0; i < CHESSBOARD_SIZE * CHESSBOARD_SIZE; i++) {
    p_noise_[i] /= deno;
  }
}