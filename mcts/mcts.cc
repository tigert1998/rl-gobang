#include "mcts.h"

#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <random>

void MCTS::EnsureRoot() {
  if (root_ == nullptr) {
    root_.reset(new MCTSNode(chessboard_, nullptr));
    if (root_->evaluating()) {
      root_->inc_vloss_cnt();
      task_queue_.PushBack(root_.get());
      DispatchBatchInference();
    }
  }
}

MCTS::MCTS(const Chessboard& chessboard, double vloss, int batch_size,
           const PolicyCallback& policy)
    : chessboard_(chessboard),
      policy_(policy),
      root_(nullptr),
      vloss_(vloss),
      batch_size_(batch_size) {}

void MCTS::Search(int num_sims, double cpuct, double dirichlet_alpha) {
  EnsureRoot();

  if (dirichlet_alpha > 0) {
    AllocateNoise(dirichlet_alpha);
    root_->set_p_noise(p_noise_);
  }

  for (int i = 0; i < num_sims; i++) {
    Simulate(cpuct);
  }

  DispatchBatchInference();
  CheckVlossCnt(root_.get());
}

void MCTS::Simulate(double cpuct) {
  bool expanded = false;

  auto node = root_.get();
  node->inc_vloss_cnt();

  for (;;) {
    if (node->terminated()) {
      BackupFromLeaf(node);
      break;
    } else if (node->evaluating() && expanded) {
      // expand a new node
      if (task_queue_.Size() >= batch_size_) {
        DispatchBatchInference();
      }
      task_queue_.PushBack(node);
      break;
    } else if (node->evaluating() && !expanded) {
      // previous evaluating node
      DispatchBatchInference();
      assert(!node->evaluating());
    }

    auto xy = node->Select(cpuct, vloss_);
    expanded = node->Expand(xy.first, xy.second);
    node = node->child(xy.first, xy.second);
    node->inc_vloss_cnt();
  }
}

void MCTS::StepForward(int x, int y) {
  chessboard_ = root_->child(x, y)->chessboard();
  root_ = root_->child_ownership(x, y);
  root_->set_father(nullptr);
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
  std::gamma_distribution<double> g(alpha, 1);
  static std::default_random_engine rng(std::time(nullptr));

  double deno = 0;

  for (int i = 0; i < CHESSBOARD_SIZE * CHESSBOARD_SIZE; i++) {
    p_noise_[i] = g(rng);
    deno += p_noise_[i];
  }
  for (int i = 0; i < CHESSBOARD_SIZE * CHESSBOARD_SIZE; i++) {
    p_noise_[i] /= deno;
  }
}

void MCTS::DispatchBatchInference() {
  constexpr int LEN = CHESSBOARD_SIZE * CHESSBOARD_SIZE;
  thread_local char* chessboards_buf[LEN];
  thread_local double* probs_buf[LEN];
  thread_local double* vs_buf[LEN];

  for (int batch_id = 0;
       batch_id < (task_queue_.Size() + batch_size_ - 1) / batch_size_;
       batch_id++) {
    int from = batch_id * batch_size_ + task_queue_.front();
    int to = std::min(from + batch_size_ - 1, task_queue_.rear());
    int n = to - from + 1;

    for (int i = 0; i < n; i++) {
      auto node = task_queue_[i + from];
      chessboards_buf[i] = node->chessboard_.Data();
      probs_buf[i] = node->p_;
      vs_buf[i] = &node->v_;
    }
    policy_(n, chessboards_buf, probs_buf, vs_buf);
  }

  for (int i = task_queue_.front(); i <= task_queue_.rear(); i++) {
    BackupFromLeaf(task_queue_[i]);
  }
  task_queue_.Clear();
}

void MCTS::BackupFromLeaf(MCTSNode* node) {
  double delta_v = node->v();
  while (node != nullptr) {
    node->Backup(delta_v);
    node->dec_vloss_cnt();

    node = node->father();
    delta_v = -delta_v;
  }
}

void MCTS::CheckVlossCnt(MCTSNode* node) {
  if (node->vloss_cnt_ != 0) {
    puts("node->vloss_cnt_ != 0");
    exit(1);
  }
  for (int i = 0; i < CHESSBOARD_SIZE * CHESSBOARD_SIZE; i++) {
    auto new_node = node->childs_[i].get();
    if (new_node == nullptr) {
      continue;
    }
    CheckVlossCnt(new_node);
  }
}