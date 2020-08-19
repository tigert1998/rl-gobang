#ifndef MCTS_STATIC_QUEUE_H_
#define MCTS_STATIC_QUEUE_H_

template <typename T, int N>
class StaticQueue {
 public:
  StaticQueue() { Clear(); }

  virtual void PushBack(const T &o) { arr_[++rear_] = o; }

  template <typename... Args>
  void EmplaceBack(Args &&... args) {
    arr_[++rear_] = T(std::forward<Args>(args)...);
  }

  int front() const { return front_; }
  int rear() const { return rear_; }
  int Size() const { return rear_ - front_ + 1; }

  T &operator[](int i) { return arr_[i]; }

  void Clear() {
    front_ = 0;
    rear_ = -1;
  }

 private:
  int front_, rear_;
  T arr_[N];
};

#endif