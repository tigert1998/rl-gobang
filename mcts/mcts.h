#ifndef MCTS_MCTS_H_
#define MCTS_MCTS_H_

#include "chessboard.h"
#include "mcts_node.h"
#include "static_queue.h"

class MCTS {
 public:
  using PolicyCallback = std::function<void(int n, char** chessboards,
                                            double** probs, double** vs)>;

  MCTS(const Chessboard& chessboard, double vloss, int batch_size,
       const PolicyCallback& policy);

  void Search(int num_sims, double cpuct, double dirichlet_alpha);

  void StepForward(int x, int y);

  bool terminated();

  void chessboard(char* ptr);

  void GetPi(double temperature, double* out);

  double v();

 private:
  Chessboard chessboard_;
  PolicyCallback policy_;
  double p_noise_[CHESSBOARD_SIZE * CHESSBOARD_SIZE];
  std::unique_ptr<MCTSNode> root_;
  StaticQueue<MCTSNode*, CHESSBOARD_SIZE * CHESSBOARD_SIZE> task_queue_;

  double vloss_;
  int batch_size_;

  void Simulate(double cpuct);

  void BackupFromLeaf(MCTSNode* node);

  void DispatchBatchInference();

  void EnsureRoot();

  void AllocateNoise(double alpha);
};

#endif