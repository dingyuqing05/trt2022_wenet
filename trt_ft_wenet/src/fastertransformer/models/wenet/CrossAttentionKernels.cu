/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/kernels/bfloat16_fallback_kenrels.cuh"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/models/wenet/WenetKernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
__global__ void addQKVBiasTranspose(T* q_out,
                                    T* k_out,
                                    T* v_out,
                                    const T* __restrict q_in,
                                    const T* __restrict bias_q,
                                    const T* __restrict k_in,
                                    const T* __restrict bias_k,
                                    const T* __restrict v_in,
                                    const T* __restrict bias_v,
                                    const int batch_size,
                                    const int seq_len1,
                                    const int seq_len2,
                                    const int head_num,
                                    const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id1 = batch_id * seq_len1 + word_id;
    const int row_id2 = batch_id * seq_len2 + word_id;

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id1 = batch_id * (head_num * seq_len1 * size_per_head) + head_id * seq_len1 * size_per_head
                               + word_id * size_per_head + size_id;
        const int target_id2 = batch_id * (head_num * seq_len2 * size_per_head) + head_id * seq_len2 * size_per_head
                               + word_id * size_per_head + size_id;

        const int src_id1 = row_id1 * n + col_id;
        const int src_id2 = row_id2 * n + col_id;

        if (word_id < seq_len1)
            q_out[target_id1] = __ldg(&q_in[src_id1]) + __ldg(&bias_q[col_id]);

        if (word_id < seq_len2) {
            k_out[target_id2] = __ldg(&k_in[src_id2]) + __ldg(&bias_k[col_id]);

            v_out[target_id2] = __ldg(&v_in[src_id2]) + __ldg(&bias_v[col_id]);
        }
    }
}

template<typename T>
__global__ void QKVTranspose(T* q_out,
                             T* k_out,
                             T* v_out,
                             const T* __restrict q_in,
                             const T* __restrict k_in,
                             const T* __restrict v_in,
                             const int batch_size,
                             const int seq_len1,
                             const int seq_len2,
                             const int head_num,
                             const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id1 = batch_id * seq_len1 + word_id;
    const int row_id2 = batch_id * seq_len2 + word_id;

    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id1 = batch_id * (head_num * seq_len1 * size_per_head) + head_id * seq_len1 * size_per_head
                               + word_id * size_per_head + size_id;
        const int target_id2 = batch_id * (head_num * seq_len2 * size_per_head) + head_id * seq_len2 * size_per_head
                               + word_id * size_per_head + size_id;

        const int src_id1 = row_id1 * n + col_id;
        const int src_id2 = row_id2 * n + col_id;

        if (word_id < seq_len1)
            q_out[target_id1] = __ldg(&q_in[src_id1]);

        if (word_id < seq_len2) {
            k_out[target_id2] = __ldg(&k_in[src_id2]);
            v_out[target_id2] = __ldg(&v_in[src_id2]);
        }
    }
}

template<typename T>
void invokeAddQKVBiasTranspose(T* q_buf,
                               T* k_buf,
                               T* v_buf,
                               T* Q,
                               const T* bias_Q,
                               T* K,
                               const T* bias_K,
                               T* V,
                               const T* bias_V,
                               const int batch_size,
                               const int seq_len1,
                               const int seq_len2,
                               const int head_num,
                               const int size_per_head,
                               cudaStream_t stream)
{
    const int k = head_num * size_per_head;
    int seq_len12 = max(seq_len1, seq_len2);
    dim3 grid(batch_size, seq_len12);
    bool is_add_bias = bias_Q != nullptr;
    if (sizeof(T) == 4 || k % 2 != 0) {
        dim3 block(min(k, 512));
        if (is_add_bias) {
            addQKVBiasTranspose<T><<<grid, block, 0, stream>>>(q_buf,
                                                               k_buf,
                                                               v_buf,
                                                               Q,
                                                               bias_Q,
                                                               K,
                                                               bias_K,
                                                               V,
                                                               bias_V,
                                                               batch_size,
                                                               seq_len1,
                                                               seq_len2,
                                                               head_num,
                                                               size_per_head);
        }
        else {
            QKVTranspose<T><<<grid, block, 0, stream>>>(
                q_buf, k_buf, v_buf, Q, K, V, batch_size, seq_len1, seq_len2, head_num, size_per_head);
        }
        sync_check_cuda_error();
    }
    else {
        dim3 block(min(k / 2, 512));
        if (is_add_bias) {
            addQKVBiasTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                                   (half2*)k_buf,
                                                                   (half2*)v_buf,
                                                                   (const half2*)Q,
                                                                   (const half2*)bias_Q,
                                                                   (const half2*)K,
                                                                   (const half2*)bias_K,
                                                                   (const half2*)V,
                                                                   (const half2*)bias_V,
                                                                   batch_size,
                                                                   seq_len1,
                                                                   seq_len2,
                                                                   head_num,
                                                                   size_per_head / 2);
        }
        else {
            QKVTranspose<half2><<<grid, block, 0, stream>>>((half2*)q_buf,
                                                            (half2*)k_buf,
                                                            (half2*)v_buf,
                                                            (const half2*)Q,
                                                            (const half2*)K,
                                                            (const half2*)V,
                                                            batch_size,
                                                            seq_len1,
                                                            seq_len2,
                                                            head_num,
                                                            size_per_head / 2);
        }
        sync_check_cuda_error();
    }
}

template void invokeAddQKVBiasTranspose(float* q_buf,
                                        float* k_buf,
                                        float* v_buf,
                                        float* Q,
                                        const float* bias_Q,
                                        float* K,
                                        const float* bias_K,
                                        float* V,
                                        const float* bias_V,
                                        const int batch_size,
                                        const int seq_len1,
                                        const int seq_len2,
                                        const int head_num,
                                        const int size_per_head,
                                        cudaStream_t stream);

template void invokeAddQKVBiasTranspose(half* q_buf,
                                        half* k_buf,
                                        half* v_buf,
                                        half* Q,
                                        const half* bias_Q,
                                        half* K,
                                        const half* bias_K,
                                        half* V,
                                        const half* bias_V,
                                        const int batch_size,
                                        const int seq_len1,
                                        const int seq_len2,
                                        const int head_num,
                                        const int size_per_head,
                                        cudaStream_t stream);

// TODO(bhsueh) Rename the softmax_kernel_v4 to softmax_kernel
template<int ITEMS_PER_THREAD, typename T, typename T_IN>
__global__ void softmax_kernel_v4(T* qk_buf_,
                                  const T_IN* qk_buf_src,
                                  const T* attr_mask,
                                  const int batch_size,
                                  const int head_num,
                                  const int seq_len1,
                                  const int seq_len2,
                                  const T scalar)
{
    for (int seq_id = blockIdx.x; seq_id < seq_len1; seq_id += gridDim.x) {
        float data[ITEMS_PER_THREAD];
        int qk_offset;
        __shared__ float s_mean, s_max;
        float local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len2; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len1 + seq_id) * seq_len2 + blockDim.x * i + threadIdx.x;
            int mask_offset = (blockIdx.y * seq_len1 + seq_id) * seq_len2 + blockDim.x * i + threadIdx.x;

            float qk = static_cast<float>(qk_buf_src[qk_offset]);

            float mask_val = static_cast<float>(ldg(&attr_mask[mask_offset]));

            mask_val = (1.0f - mask_val) * -10000.0f;

            data[i] = qk * static_cast<float>(scalar) + mask_val;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len2; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }
        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < seq_len2; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len1 + seq_id) * seq_len2 + blockDim.x * i + threadIdx.x;
            qk_buf_[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}

template<typename T, int ITEMS_PER_THREAD>
__global__ void softmax_kernel_v4_half2(T* qk_buf_,
                                        const T* attr_mask,
                                        const int batch_size,
                                        const int head_num,
                                        const int seq_len1,
                                        const int seq_len2,
                                        const T scalar)
{
    using T2 = typename TypeConverter<T>::Type;
    T2* qk_buf_half2 = (T2*)qk_buf_;
    const T2* attr_mask_half2 = (const T2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len1; seq_id += gridDim.x) {
        T2 data[ITEMS_PER_THREAD];
        int qk_offset;
        __shared__ float s_mean, s_max;
        float local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len2 / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len1 + seq_id) * (seq_len2 / 2) + blockDim.x * i
                        + threadIdx.x;
            int mask_offset = (blockIdx.y * seq_len1 + seq_id) * (seq_len2 / 2) + blockDim.x * i + threadIdx.x;

            T2 qk = qk_buf_half2[qk_offset];
            T2 mask_val = ldg(&attr_mask_half2[mask_offset]);
            mask_val = hmul2<T2>(hsub2<T2>(float2type2<T2>(1.0f), mask_val), float2type2<T2>(-10000.0f));

            data[i] = hadd2<T2>(hmul2<T2>(qk, type2type2<T, T2>(scalar)), mask_val);

            local_max = fmax(local_max, fmax((float)data[i].x, (float)data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len2 / 2) && i < ITEMS_PER_THREAD; i++) {
            data[i] = hexp2<T2>(hsub2<T2>(data[i], float2type2<T2>(s_max)));
            local_sum += (float)(data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len2 / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len1 + seq_id) * (seq_len2 / 2) + blockDim.x * i
                        + threadIdx.x;
            qk_buf_half2[qk_offset] = hmul2<T2>(data[i], float2type2<T2>(s_mean));
        }
    }
}

template<typename T, int ITEMS_PER_THREAD, int NUM>
__global__ void softmax_kernel_v5_half2(T* qk_buf_,
                                        const T* attr_mask,
                                        const int batch_size,
                                        const int head_num,
                                        const int seq_len1,
                                        const int seq_len2,
                                        const T scalar)
{
    using T2 = typename TypeConverter<T>::Type;
    T2* qk_buf_half2 = (T2*)qk_buf_;

    const T2* attr_mask_half2 = (const T2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len1; seq_id += gridDim.x * NUM) {
        T2 data[NUM][ITEMS_PER_THREAD];

        int qk_offset[NUM];

        __shared__ float s_sum[NUM], s_max[NUM];
        float local_max[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_max[j] = -1e20f;
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len2 / 2) && i < ITEMS_PER_THREAD; i++) {
            int mask_offset[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] =
                    ((blockIdx.y * head_num + blockIdx.z) * seq_len1 + seq_id + j * gridDim.x) * (seq_len2 / 2)
                    + blockDim.x * i + threadIdx.x;
                mask_offset[j] =
                    (blockIdx.y * seq_len1 + seq_id + j * gridDim.x) * (seq_len2 / 2) + blockDim.x * i + threadIdx.x;
            }

            T2 mask_val[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = ldg(&attr_mask_half2[mask_offset[j]]);
            }

            T2 qk[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk[j] = qk_buf_half2[qk_offset[j]];
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = hmul2<T2>(hsub2<T2>(float2type2<T2>(1.0f), mask_val[j]), float2type2<T2>(-10000.0f));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = hadd2<T2>(hmul2<T2>(qk[j], type2type2<T, T2>(scalar)), mask_val[j]);
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }

        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, NUM>(local_max);
        }
        else {
            blockReduceMaxV2<float, NUM>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len2 / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = hexp2<T2>(hsub2<T2>(data[j][i], float2type2<T2>(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, NUM>(local_sum);
        }
        else {
            blockReduceSumV2<float, NUM>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len2 / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] =
                    ((blockIdx.y * head_num + blockIdx.z) * seq_len1 + seq_id + j * gridDim.x) * (seq_len2 / 2)
                    + blockDim.x * i + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_buf_half2[qk_offset[j]] = hmul2<T2>(data[j][i], float2type2<T2>(s_sum[j]));
            }
        }
    }
}

#define SOFTMAX_KERNEL(ITEMS_PER_THREAD)                                                                               \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
        if (grid.x % 4 == 0) {                                                                                         \
            grid.x /= 4;                                                                                               \
            softmax_kernel_v5_half2<half, ITEMS_PER_THREAD, 4><<<grid, block, 0, stream>>>(                            \
                (half*)buffer, (const half*)attr_mask, batch_size, head_num, seq_len1, seq_len2, (const half)scalar);  \
        }                                                                                                              \
        else {                                                                                                         \
            softmax_kernel_v4_half2<half, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>(                               \
                (half*)buffer, (const half*)attr_mask, batch_size, head_num, seq_len1, seq_len2, (const half)scalar);  \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        softmax_kernel_v4<ITEMS_PER_THREAD, T, T_IN><<<grid, block, 0, stream>>>(                                      \
            buffer, buffer_src, attr_mask, batch_size, head_num, seq_len1, seq_len2, scalar);                          \
    }

#ifdef ENABLE_BF16
#define SOFTMAX_KERNEL_BF16(ITEMS_PER_THREAD)                                                                          \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
        if (grid.x % 4 == 0) {                                                                                         \
            grid.x /= 4;                                                                                               \
            softmax_kernel_v5_half2<__nv_bfloat16, ITEMS_PER_THREAD, 4>                                                \
                <<<grid, block, 0, stream>>>((__nv_bfloat16*)buffer,                                                   \
                                             (const __nv_bfloat16*)attr_mask,                                          \
                                             batch_size,                                                               \
                                             head_num,                                                                 \
                                             seq_len1,                                                                 \
                                             seq_len2,                                                                 \
                                             (const __nv_bfloat16)scalar);                                             \
        }                                                                                                              \
        else {                                                                                                         \
            softmax_kernel_v4_half2<__nv_bfloat16, ITEMS_PER_THREAD>                                                   \
                <<<grid, block, 0, stream>>>((__nv_bfloat16*)buffer,                                                   \
                                             (const __nv_bfloat16*)attr_mask,                                          \
                                             batch_size,                                                               \
                                             head_num,                                                                 \
                                             seq_len1,                                                                 \
                                             seq_len2,                                                                 \
                                             (const __nv_bfloat16)scalar);                                             \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        softmax_kernel_v4<ITEMS_PER_THREAD, __nv_bfloat16, T_IN><<<grid, block, 0, stream>>>(                          \
            buffer, buffer_src, attr_mask, batch_size, head_num, seq_len1, seq_len2, scalar);                          \
    }
#endif  // ENABLE_BF16

template<typename T, typename T_IN>
void invokeMaskedSoftMax(T* buffer,
                         const T_IN* buffer_src,
                         const T* attr_mask,
                         const int batch_size,
                         const int seq_len1,
                         const int seq_len2,
                         const int head_num,
                         const T scalar,
                         cudaStream_t stream)
{

    dim3 grid(seq_len1, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len1) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && seq_len2 % 2 == 0;
    dim3 block((seq_len2 / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL(1)
    }
    else {
        FT_CHECK(seq_len2 <= 4096);
    }
}

#ifdef ENABLE_BF16
template<>
void invokeMaskedSoftMax(__nv_bfloat16* buffer,
                         const __nv_bfloat16* buffer_src,
                         const __nv_bfloat16* attr_mask,
                         const int batch_size,
                         const int seq_len1,
                         const int seq_len2,
                         const int head_num,
                         const __nv_bfloat16 scalar,
                         cudaStream_t stream)
{

    using T_IN = __nv_bfloat16;
    dim3 grid(seq_len1, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len1) / 32.0f);
    }

    bool is_half2 = seq_len2 % 2 == 0;
    dim3 block((seq_len2 / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL_BF16(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL_BF16(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL_BF16(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL_BF16(1)
    }
    else {
        FT_CHECK(seq_len2 <= 4096);
    }
}

template<>
void invokeMaskedSoftMax(__nv_bfloat16* buffer,
                         const float* buffer_src,
                         const __nv_bfloat16* attr_mask,
                         const int batch_size,
                         const int seq_len1,
                         const int seq_len2,
                         const int head_num,
                         const __nv_bfloat16 scalar,
                         cudaStream_t stream)
{
    using T_IN = float;
    dim3 grid(seq_len1, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len1) / 32.0f);
    }

    bool is_half2 = false;
    dim3 block((seq_len2 / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL_BF16(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL_BF16(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL_BF16(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL_BF16(1)
    }
    else {
        FT_CHECK(seq_len2 <= 4096);
    }
}
#endif  // ENABLE_BF16

template void invokeMaskedSoftMax(float* buffer,
                                  const float* buffer_src,
                                  const float* attr_mask,
                                  const int batch_size,
                                  const int seq_len1,
                                  const int seq_len2,
                                  const int head_num,
                                  const float scalar,
                                  cudaStream_t stream);

template void invokeMaskedSoftMax(half* buffer,
                                  const float* buffer_src,
                                  const half* attr_mask,
                                  const int batch_size,
                                  const int seq_len1,
                                  const int seq_len2,
                                  const int head_num,
                                  const half scalar,
                                  cudaStream_t stream);

template void invokeMaskedSoftMax(half* buffer,
                                  const half* buffer_src,
                                  const half* attr_mask,
                                  const int batch_size,
                                  const int seq_len1,
                                  const int seq_len2,
                                  const int head_num,
                                  const half scalar,
                                  cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeMaskedSoftMax(__nv_bfloat16* buffer,
                                  const __nv_bfloat16* buffer_src,
                                  const __nv_bfloat16* attr_mask,
                                  const int batch_size,
                                  const int seq_len1,
                                  const int seq_len2,
                                  const int head_num,
                                  const __nv_bfloat16 scalar,
                                  cudaStream_t stream);

template void invokeMaskedSoftMax(__nv_bfloat16* buffer,
                                  const float* buffer_src,
                                  const __nv_bfloat16* attr_mask,
                                  const int batch_size,
                                  const int seq_len1,
                                  const int seq_len2,
                                  const int head_num,
                                  const __nv_bfloat16 scalar,
                                  cudaStream_t stream);
#endif  // ENABLE_BF16

}  // namespace fastertransformer
