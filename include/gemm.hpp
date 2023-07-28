
#ifndef GEMM_HPP
#define GEMM_HPP

#include <cstddef>
#include <cstdint>

template <uint32_t M_, uint32_t K_, uint32_t N_, uint32_t group_tile_m_,
          uint32_t group_tile_n_, uint32_t stride_k_, uint32_t sg_tile_m_,
          uint32_t sg_tile_n_>
struct Config {
  constexpr static uint32_t M = M_;
  constexpr static uint32_t N = N_;
  constexpr static uint32_t K = K_;
  constexpr static uint32_t group_tile_m = group_tile_m_;
  constexpr static uint32_t group_tile_n = group_tile_n_;
  constexpr static uint32_t stride_k = stride_k_;
  constexpr static uint32_t sg_tile_m = sg_tile_m_;
  constexpr static uint32_t sg_tile_n = sg_tile_n_;
};

struct HostTag {};
struct NaiveGPUTag {};
struct NaiveGPUWithESIMDTag{};
struct GPUTag {};

template <typename T, typename Config, typename Tag>
class GEMM;

#endif
