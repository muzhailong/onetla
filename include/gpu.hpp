#ifndef GPU_HPP
#define GPU_HPP

#include <cstdint>

#include <sycl/ext/intel/esimd.hpp>

#include <sycl/sycl.hpp>

#include "gemm.hpp"

template <typename T, typename Config>
class GEMM<T, Config, GPUTag> {
public:
  void operator()(sycl::queue& q, T* C, T* A, T* B) {

    q.submit([&](sycl::handler& h) {
      h.parallel_for(
          sycl::nd_range<2>{
              sycl::range<2>{Config::M / Config::sg_tile_m,
                             Config::N / Config::sg_tile_n},
              sycl::range<2>{Config::group_tile_m / Config::sg_tile_m,
                             Config::group_tile_n / Config::sg_tile_n}},
          [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            namespace esimd = sycl::ext::intel::esimd;
            using namespace sycl::ext::oneapi::experimental;
            esimd::slm_init<(Config::group_tile_m + Config::group_tile_n) *
                            Config::stride_k * sizeof(T) * 2>();

            uint32_t a_slm = 0;
            uint32_t b_slm =
                Config::group_tile_m * Config::stride_k * sizeof(T);
            uint32_t a_slm1 =
                b_slm + Config::group_tile_n * Config::stride_k * sizeof(T);
            uint32_t b_slm1 =
                a_slm1 + Config::group_tile_m * Config::stride_k * sizeof(T);

            const uint32_t BLOCK_SIZE = 32;
            const uint32_t offset_x = it.get_local_id(0);
            const uint32_t offset_y = it.get_local_id(1);

            const uint32_t global_x = it.get_global_id(0);
            const uint32_t global_y = it.get_global_id(1);

            const uint32_t group_x_size =
                Config::group_tile_m / Config::sg_tile_m;
            const uint32_t group_y_size =
                Config::group_tile_n / Config::sg_tile_n;

            const uint32_t group_x = global_x / group_x_size * group_x_size;
            const uint32_t group_y = global_y / group_y_size * group_y_size;

            const uint32_t begin_m = group_x * group_x_size * Config::sg_tile_m;
            const uint32_t begin_n = group_y * group_y_size * Config::sg_tile_n;
            // 1. load global memory into local memory
            // we need to split task into multi subgroup(esimd)

            // for a sg
            // in a tile, need to load a block [sg_tile_m, K / group_y_size]
            // offset is: (group_x * sg_tile_m, group_y * K / group_y_size)
            // in b tile, need to load a block [K / group_x_size, sg_tile_n]
            // offset is: (group_x * K / group_x_size, group_y * sg_tile_n)
            // need to use gather and scatter
            // gather: global memory -> register
            // scatter: register -> local memory
            const uint32_t a_tile_m = Config::sg_tile_m;
            const uint32_t a_tile_n = Config::K / group_y_size;
            const uint32_t b_tile_m = Config::K / group_x_size;
            const uint32_t b_tile_n = Config::sg_tile_n;

            // group level work
            // need the follwing information:
            // 1. acc reg and offset
            // 2. slm_begin_m, slm_end_m
            // 3. slm_begin_n, slm_end_n
            esimd::simd<T, Config::sg_tile_m* Config::sg_tile_n> acc = T();
            auto acc_2d = acc.template bit_cast_view<T, Config::sg_tile_m,
                                                     Config::sg_tile_n>();
            // global_x * sg_tile_m, global_y* sg_tile_n
            // task per subgroup
            const uint32_t m_num_elements_per_sg =
                group_x_size * Config::sg_tile_m * Config::stride_k /
                (group_x_size * group_y_size);
            const uint32_t n_num_elements_per_sg =
                group_y_size * Config::sg_tile_n * Config::stride_k /
                (group_x_size * group_y_size);

            const uint32_t group_linear_id = offset_x * group_y_size + offset_y;
            const uint32_t slm_m_offset =
                group_linear_id * m_num_elements_per_sg * sizeof(T);
            const uint32_t slm_n_offset =
                (offset_y * group_y_size + offset_x) * n_num_elements_per_sg * sizeof(T);

            // prefetch, global memory -> shared local memory
            esimd::simd<uint32_t, m_num_elements_per_sg>
                m_tile_offset;  // 相对于当前顶点位置的偏移 row major
            esimd::simd<uint32_t, n_num_elements_per_sg>
                n_tile_offset;  // row major
            {
              for (uint32_t i = 0; i < m_num_elements_per_sg; ++i) {
                const uint32_t tmp = group_linear_id * m_num_elements_per_sg;
                const uint32_t tx = tmp / Config::stride_k;
                const uint32_t ty = tmp % Config::stride_k;
                m_tile_offset[i] = (tx * Config::K + ty) * sizeof(T);
              }

              auto n_tile_offset_2d = n_tile_offset.template bit_cast_view<
                  T, n_num_elements_per_sg / Config::sg_tile_n,
                  Config::sg_tile_n>();

              for (uint32_t i = 0; i < Config::sg_tile_n; ++i) {
                const uint32_t tmp = n_num_elements_per_sg / Config::sg_tile_n;
                n_tile_offset_2d.row(0)[i] = (offset_x * tmp * Config::N +
                                              offset_y * Config::sg_tile_n) *
                                             sizeof(T);
              }
              for (uint32_t i = 1;
                   i < n_num_elements_per_sg / Config::sg_tile_n; ++i) {
                n_tile_offset_2d.row(i) =
                    n_tile_offset_2d.row(i - 1) + Config::N * sizeof(T);
              }

              auto a_tmp = esimd::gather<T, m_num_elements_per_sg>(
                  A + begin_m * Config::K, m_tile_offset);
              esimd::slm_block_store(a_slm + slm_m_offset, a_tmp);

              auto b_tmp = esimd::gather<T, n_num_elements_per_sg>(
                  B + begin_n, n_tile_offset);
              esimd::slm_block_store(b_slm + slm_n_offset, b_tmp);
            }

            // main loop
            for (uint32_t k = 0, ki = 0; k < Config::K;
                 k += Config::stride_k, ++ki) {
              esimd::barrier();
              if (k != Config::K - Config::stride_k) {
                // load global memory -> slm1
                auto a_tmp = esimd::gather<T, m_num_elements_per_sg>(
                    A + begin_m * Config::K + k * Config::stride_k,
                    m_tile_offset);
                esimd::slm_block_store((ki & 1) ? a_slm : a_slm1 + slm_m_offset,
                                       a_tmp);
                auto b_tmp = esimd::gather<T, n_num_elements_per_sg>(
                    B + begin_n + k * Config::stride_k * Config::N,
                    n_tile_offset);
                esimd::slm_block_store((ki & 1) ? b_slm : b_slm1 + slm_n_offset,
                                       b_tmp);
              }

              // compute, here in subgroup level
              // use double register buffer
              auto a_buf = esimd::slm_block_load<T, Config::sg_tile_m *
                                                        Config::stride_k>(
                  ((ki & 1) ? a_slm1 : a_slm) + slm_m_offset);
              auto b_buf = esimd::slm_block_load<T, Config::sg_tile_m *
                                                        Config::stride_k>(
                  ((ki & 1) ? b_slm1 : b_slm) + slm_n_offset);

              auto a_buf_2d = a_buf.template bit_cast_view<T, Config::sg_tile_m,
                                                           Config::stride_k>();
              auto b_buf_2d = b_buf.template bit_cast_view<T, Config::stride_k,
                                                           Config::sg_tile_n>();
              for (uint32_t i = 0; i < Config::sg_tile_m; ++i) {
                esimd::simd<T, Config::sg_tile_n> acc_row = T();
                for (uint32_t j = 0; j < Config::stride_k; ++j) {
                  acc_row += b_buf_2d.row(j) * a_buf_2d.row(i)[j];
                }
                acc_2d.row(i) += acc_row;
              }
            }

            esimd::simd<uint32_t, Config::sg_tile_m * Config::sg_tile_n>
                acc_offset;
            {
              auto acc_offset_2d =
                  acc_offset.template bit_cast_view<T, Config::sg_tile_m,
                                                    Config::sg_tile_n>();
              for (uint32_t i = 0; i < Config::sg_tile_n; ++i) {
                acc_offset_2d.row(0)[i] =
                    (global_x * Config::sg_tile_m * Config::N +
                     global_y * Config::sg_tile_n + i) *
                    sizeof(T);
              }
              for (uint32_t i = 1; i < Config::sg_tile_n; ++i) {
                acc_offset_2d.row(i) =
                    acc_offset_2d.row(i - 1) + Config::N * sizeof(T);
              }
            }

            for (uint32_t i = 0; i < Config::sg_tile_m * Config::sg_tile_n;
                 i += BLOCK_SIZE) {
              auto val = acc.template select<BLOCK_SIZE, 1>(i);
              auto offset = acc_offset.template select<BLOCK_SIZE, 1>(i);
              esimd::scatter<T, BLOCK_SIZE>(C, offset, val);
            }
          });
    });
  }
};

#endif
