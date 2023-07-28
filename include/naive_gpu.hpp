#ifndef NAIVE_GPU_HPP
#define NAIVE_GPU_HPP

#include "gemm.hpp"

#include <array>

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

template <typename T, typename Config> class GEMM<T, Config, NaiveGPUTag> {
public:
  void operator()(sycl::queue &q, T *C, T *A, T *B) {
    // basic range
    q.submit([&](sycl::handler &h) {
      // sycl::stream out(1024, 256, h);
      h.parallel_for(sycl::nd_range<2>{sycl::range<2>{Config::M, Config::N},
                                       sycl::range<2>{Config::group_tile_m,
                                                      Config::group_tile_n}},
                     [=](sycl::nd_item<2> it) {
                       uint32_t i = it.get_global_id(0);
                       uint32_t j = it.get_global_id(1);
                       T sm = T();
                       for (uint32_t f = 0; f < Config::K; ++f) {
                         sm += A[i * Config::K + f] * B[f * Config::N + j];
                       }
                       C[i * Config::N + j] = sm;
                     });
    });
  }
};

template <typename T, typename Config>
class GEMM<T, Config, NaiveGPUWithESIMDTag> {
public:
  void operator()(sycl::queue &q, T *C, T *A, T *B) {
    // basic range
    namespace esimd = sycl::ext::intel::esimd;
    q.submit([&](sycl::handler &h) {
      h.parallel_for(
          sycl::nd_range<2>{
              sycl::range<2>{Config::M / Config::sg_tile_m,
                             Config::N / Config::sg_tile_n},
              sycl::range<2>{Config::group_tile_m / Config::sg_tile_m,
                             Config::group_tile_n / Config::sg_tile_n}},
          [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            // https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd
            uint32_t x = it.get_global_id(0);
            uint32_t y = it.get_global_id(1);

            using global_ptr =
                sycl::multi_ptr<T, sycl::access::address_space::global_space>;

            uint32_t begin_m = x * Config::sg_tile_m;
            // uint32_t end_m = (i + 1) * Config::sg_tile_m <= Config::M
            //                      ? (j + 1) * Config::sg_tile_m
            //                      : Config::M;  // [begin_m, end_m)

            uint32_t begin_n = y * Config::sg_tile_n;
            // uint32_t end_n = (i + 1) * Config::sg_tile_n <= Config::N
            //                      ? (i + 1) * Config::sg_tile_n
            //                      : Config::N;  // [begin_n, end_n)

            esimd::simd<T, Config::sg_tile_m *Config::sg_tile_n> res =
                T(); // register
            auto res_2d = res.template bit_cast_view<T, Config::sg_tile_m,
                                                     Config::sg_tile_n>();

            esimd::simd<uint32_t, Config::sg_tile_m * Config::stride_k>
                m_offset;
            esimd::simd<uint32_t, Config::stride_k * Config::sg_tile_n>
                n_offset;
            esimd::simd<uint32_t, Config::sg_tile_m * Config::sg_tile_n>
                res_offset;

            for (uint32_t i = 0; i < Config::sg_tile_m; ++i) {
              for (uint32_t j = 0; j < Config::stride_k; ++j) {
                m_offset[i * Config::stride_k + j] =
                    (i * Config::K + j) * sizeof(T);
              }
            }
            for (uint32_t i = 0; i < Config::stride_k; ++i) {
              for (uint32_t j = 0; j < Config::sg_tile_n; ++j) {
                n_offset[i * Config::sg_tile_n + j] =
                    (i * Config::N + j) * sizeof(T);
              }
            }
            for (uint32_t i = 0; i < Config::sg_tile_m; ++i) {
              for (uint32_t j = 0; j < Config::sg_tile_n; ++j) {
                res_offset[i * Config::sg_tile_n + j] =
                    (i * Config::N + j) * sizeof(T);
              }
            }

            for (uint32_t ki = 0; ki < Config::K; ki += Config::stride_k) {
              // 1. load tile n (block)
              //  address: ki * N + begin_n;
              auto n_reg =
                  esimd::gather<T, Config::sg_tile_n * Config::stride_k>(
                      B + ki * Config::N + begin_n, n_offset);
              // 2. load tile m (gather)
              //  address: begin_m * K + ki
              auto m_reg =
                  esimd::gather<T, Config::sg_tile_m * Config::stride_k>(
                      A + begin_m * Config::K + ki, m_offset);
// #ifdef USE_PREFETCH
//               // prefetch tile n
//               for (uint32_t ui = 0; ui < Config::stride_k; ++ui) {
//                 global_ptr(B + (ki + ui) * Config::N + begin_n)
//                     .prefetch(Config::sg_tile_n);
//               }

//               // fetch tile m
//               for (uint32_t ui = 0; ui < Config::sg_tile_m; ++ui) {
//                 global_ptr(A + (begin_m + ui) * Config::K + ki)
//                     .prefetch(Config::stride_k);
//               }
// #endif

              it.get_sub_group().barrier();
              // 3. cross product
              auto n_reg_2d = n_reg.template bit_cast_view<T, Config::stride_k,
                                                           Config::sg_tile_n>();
              auto m_reg_2d = m_reg.template bit_cast_view<T, Config::sg_tile_m,
                                                           Config::stride_k>();

              for (uint32_t kj = 0; kj < Config::sg_tile_m; ++kj) {
                esimd::simd<T, Config::sg_tile_n> acc_row = 0;
                for (uint32_t kp = 0; kp < Config::stride_k; ++kp) {
                  acc_row += n_reg_2d.row(kp) * m_reg_2d.row(kj)[kp];
                }
                res_2d.row(kj) += acc_row;
              }
            }
            // write into memory
            // address: begin_m * N + begin_n
            // scatter num elements is 1,8,16 or 32
            // sg_tile_m * sg_tile_n % scatter_num ==0
            constexpr uint32_t SCATTER_NUM = 32;
            for (uint32_t i = 0; i < Config::sg_tile_m * Config::sg_tile_n;
                 i += SCATTER_NUM) {
              auto val = res.template select<SCATTER_NUM, 1>(i);
              auto offset = res_offset.template select<SCATTER_NUM, 1>(i);
              esimd::scatter<T, SCATTER_NUM>(C + begin_m * Config::N + begin_n,
                                             offset, val);
            }
          });
    });
  }
};

#endif
