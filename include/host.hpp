#ifndef HOST_HPP
#define HOST_HPP

#include "gemm.hpp"

#include <omp.h>
#include <sycl/sycl.hpp>

template <typename T, typename Config>
class GEMM<T, Config, HostTag> {

public:
  void operator()(T* C, T* A, T* B) {
#pragma omp parallel for
    for (size_t i = 0; i < Config::M * Config::N; ++i) {
      size_t ki = i / Config::M;
      size_t kj = i % Config::M;
      T sm = T(0);
      for (size_t f = 0; f < Config::K; ++f) {
        sm += A[ki * Config::K + f] * B[f * Config::N + kj];
      }
      C[ki * Config::N + kj] = sm;
    }
  }
};

#endif
