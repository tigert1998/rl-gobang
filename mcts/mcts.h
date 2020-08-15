#ifndef MCTS_MCTS_H_
#define MCTS_MCTS_H_

#include "chessboard.h"
#include "mcts_node.h"

class MCTS {
 public:
  using PolicyCallback = MCTSNode::PolicyCallback;

  MCTS(const Chessboard& chessboard, const PolicyCallback& policy);

  void Search(int num_sims, double cpuct);

  void StepForward(int x, int y);

  bool terminated();

  void chessboard(char* ptr);

  void GetPi(double temperature, double* out);

  double v();

 private:
  Chessboard chessboard_;
  PolicyCallback policy_;
  std::unique_ptr<MCTSNode> root_;

  void Simulate(double cpuct);

  void EnsureRoot();
};

#endif