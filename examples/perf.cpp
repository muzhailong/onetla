#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>

#include "gemm.hpp"
#include "gpu.hpp"
#include "host.hpp"
#include "naive_gpu.hpp"

template <size_t warmup_times = 1, size_t times = 10>
double perf(std::function<void(size_t)> f) {
  f(warmup_times);
  auto st = std::chrono::steady_clock::now();
  f(times);
  auto ed = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
}

template <typename T>
void init(T* h, size_t sz) {
  std::default_random_engine eg;
  std::uniform_real_distribution<T> dis(std::numeric_limits<T>::min(),
                                        std::numeric_limits<T>::max());
  for (size_t i = 0; i < sz; ++i) {
    h[i] = dis(eg);
  }
}

template <>
void init<float>(float* h, size_t sz) {
  std::default_random_engine eg;
  std::uniform_real_distribution<float> dis(.0f, 1.0f);
  for (size_t i = 0; i < sz; ++i) {
    h[i] = dis(eg);
  }
}

template <typename T>
bool equal(T* a, T* b, size_t size) {
  return std::equal(a, a + size, b);
}

template <>
bool equal<float>(float* a, float* b, size_t size) {
  return std::equal(a, a + size, b,
                    [](float x1, float x2) { return abs(x1 - x2) < 1e-2; });
}

int main() {
  constexpr size_t M = 1024 * 2;
  constexpr size_t K = 1024 * 2;
  constexpr size_t N = 1024 * 2;
  sycl::queue q(sycl::default_selector_v);

  float* A_h = sycl::malloc_host<float>(M * K, q.get_context());
  float* B_h = sycl::malloc_host<float>(K * N, q.get_context());
  float* C_h = sycl::malloc_host<float>(M * N, q.get_context());
  float* C_h_ref = sycl::malloc_host<float>(M * N, q.get_context());
  init(A_h, M * K);
  init(B_h, K * N);

  float* A_d = sycl::malloc_device<float>(M * K, q);
  float* B_d = sycl::malloc_device<float>(K * N, q);
  float* C_d = sycl::malloc_device<float>(M * N, q);
  q.memcpy(A_d, A_h, sizeof(float) * M * K);
  q.memcpy(B_d, B_h, sizeof(float) * K * N);
  q.wait_and_throw();

  using HostConfig = Config<M, K, N, 0, 0, 0, 0, 0>;
  GEMM<float, HostConfig, HostTag> host_gemm;
  auto host_fn = [&](size_t times) {
    for (size_t i = 0; i < times; ++i) {
      host_gemm(C_h_ref, A_h, B_h);
    }
  };
  double host_ms = perf<3, 10>(host_fn);
  std::cout << "host: " << host_ms << "ms\n";

  using NaiveConfig = Config<M, K, N, 32, 32, 32, 32, 16>;
  GEMM<float, NaiveConfig, NaiveGPUTag> naive_gemm;
  auto naive_fn = [&](size_t times) {
    for (size_t i = 0; i < times; ++i) {
      naive_gemm(q, C_d, A_d, B_d);
    }
    q.wait();
  };
  double naive_ms = perf<3, 10>(naive_fn);
  std::cout << "naive gpu: " << naive_ms << "ms\n";
  q.copy(C_d, C_h, M * N);
  q.wait_and_throw();
  assert("naive gpu equal" && equal(C_h, C_h_ref, M * N));

  using NaivebyEsimdConfig = Config<M, K, N, 32, 32, 4, 8, 8>;
  GEMM<float, NaivebyEsimdConfig, NaiveGPUWithESIMDTag> naive_by_esimd_gemm;
  auto naive_by_esimd_fn = [&](size_t times) {
    for (size_t i = 0; i < times; ++i) {
      naive_by_esimd_gemm(q, C_d, A_d, B_d);
    }
    q.wait();
  };
  double naive_by_esimd_ms = perf<3, 10>(naive_by_esimd_fn);
  std::cout << "naive by esimd gpu: " << naive_by_esimd_ms << "ms\n";
  q.copy(C_d, C_h, M * N);
  q.wait_and_throw();
  assert("naive gpu by esimd equal" && equal(C_h, C_h_ref, M * N));

  using GPUConfig = Config<M, K, N, 32, 32, 8, 8, 8>;
  GEMM<float, GPUConfig, GPUTag> gemm;
  auto fn = [&](size_t times) {
    for (size_t i = 0; i < times; ++i) {
      gemm(q, C_d, A_d, B_d);
    }
    q.wait_and_throw();
  };
  double ms = perf<3, 10>(fn);
  std::cout << "gpu: " << ms << "ms\n";
  q.copy(C_d, C_h, M * N);
  q.wait_and_throw();
  assert("gpu by esimd equal" && equal(C_h, C_h_ref, M * N));

  for (auto it : {A_d, B_d, C_d, A_h, B_h, C_h, C_h_ref}) {
    sycl::free(it, q.get_context());
  }
  return 0;
}
