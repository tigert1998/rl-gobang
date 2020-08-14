#include "mcts.h"

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