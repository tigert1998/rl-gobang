#ifndef MCTS_SPAN_H_
#define MCTS_SPAN_H_

#include <stdint.h>

template <typename T>
class Span {
 public:
  using size_type = size_t;

  Span(T *ptr, size_type count) : ptr_(ptr), count_(count) {}
  T *Data() { return ptr_; }
  size_type Size() { return count_; }
  T &operator[](size_type idx) { return ptr_[idx]; }

 private:
  T *ptr_;
  size_type count_;
};

#endif