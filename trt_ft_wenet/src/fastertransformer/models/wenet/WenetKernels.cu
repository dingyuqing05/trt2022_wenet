/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

namespace {
constexpr auto EPS = 1e-6f;  // 1e-5;
}
template<typename T>
__inline__ __device__ T sigmoid(T x)
{
    return T(1.0f) / (T(1.0f) + exp(-x));
}

template<typename T>
__global__ void add_bias_silu(T* out, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val = out[id];
        if (bias != nullptr) {
            val = val + ldg(&bias[id % n]);
        }
        out[id] = float(val) * sigmoid<float>(float(val));
    }
}

template<>
__global__ void add_bias_silu(half* out, const half* __restrict bias, int m, int n)
{
    half2* out_ptr = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val = out_ptr[id];
        if (bias != nullptr) {
            val = val + __ldg(&bias_ptr[id % n]);
        }
        val.x = float(val.x) * sigmoid<float>(float(val.x));
        val.y = float(val.y) * sigmoid<float>(float(val.y));
        out_ptr[id] = val;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void add_bias_silu(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, int m, int n)
{
    __nv_bfloat162* out_ptr = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val = out_ptr[id];
        if (bias != nullptr) {
            val = bf16hadd2(val, ldg(&bias_ptr[id % n]));
        }
        val.x = float(val.x) * sigmoid<float>(float(val.x));
        val.y = float(val.y) * sigmoid<float>(float(val.y));
        out_ptr[id] = val;
    }
}
#endif
template<>
__global__ void add_bias_silu(half2* out, const half2* __restrict bias, int m, int n)
{
    half2 local_bias = float2type2<half2>(0.f);
    if (threadIdx.x < n) {
        if (bias != nullptr)
            local_bias = __ldg(&bias[threadIdx.x]);
        for (int mi = 0; mi < m; mi += gridDim.x) {
            int idx = mi * n + threadIdx.x;
            half2 val = out[idx];
            val = hadd2(val, local_bias);
            val.x = float(val.x) * sigmoid<float>(float(val.x));
            val.y = float(val.y) * sigmoid<float>(float(val.y));
            out[idx] = val;
        }
    }
}

template<typename T>
void invokeAddBiasSilu(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    if (true) {
        const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
        dim3 block, grid;
        if (n / 4 / data_type_factor <= 1024) {
            block.x = n / 4 / data_type_factor;
            grid.x = m;
        }
        else {
            block.x = 1024;
            grid.x = ceil(m * n / 1024.);
        }
        add_bias_silu<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
    }
    else {
        FT_CHECK(sizeof(T) == 2);
        FT_CHECK(n % 2 == 0);
        FT_CHECK(n / 2 <= 1024);
        dim3 block, grid;
        block.x = n / 2;
        grid.x = m / 2;
        add_bias_silu<half2><<<grid, block, 0, stream>>>((half2*)out, (const half2*)bias, m, n / 2);
    }
}

template void invokeAddBiasSilu(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasSilu(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeAddBiasSilu(__nv_bfloat16* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void mask_bias_glu(
    T* out, const T* __restrict in, const T* __restrict bias, int m, int n, const T* __restrict attr_mask, int seq_len)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        int s_id = m_id % seq_len;
        int b_id = m_id / seq_len;
        T cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];
        // in = in + m_id * 2 * n;
        const T* in_ptr = in + m_id * 2 * n;

        float val1 = 0.f;
        float val2 = 0.f;

        if (cur_mask != T(0.f)) {
            val1 = in_ptr[n_id];
            val2 = in_ptr[n_id + n];
        }

        if (bias != nullptr) {
            val1 = val1 + ldg(&bias[n_id]);
            val2 = val2 + ldg(&bias[n_id + n]);
        }
        out[id] = val1 * sigmoid<float>(val2);
    }
}

template<>
__global__ void mask_bias_glu(half* out,
                              const half* __restrict in,
                              const half* __restrict bias,
                              int m,
                              int n,
                              const half* __restrict attr_mask,
                              int seq_len)
{
    half2* out_ptr = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        int s_id = m_id % seq_len;
        int b_id = m_id / seq_len;
        // const half* attr_mask = attr_mask + ;
        half cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];

        const half2* in_ptr = (half2*)in + m_id * 2 * n;
        // in = in + m_id * 2 * n;

        half2 val1 = float2type2<half2>(0.f);
        half2 val2 = float2type2<half2>(0.f);

        if (cur_mask != half(0.f)) {
            val1 = in_ptr[n_id];
            val2 = in_ptr[n_id + n];
        }

        if (bias != nullptr) {
            half2 bias1 = __ldg(&bias_ptr[n_id]);
            half2 bias2 = __ldg(&bias_ptr[n_id + n]);
            val1 = hadd2(val1, bias1);
            val2 = hadd2(val2, bias2);
        }
        half2 local_out;
        local_out.x = (float)val1.x * sigmoid<float>((float)val2.x);
        local_out.y = (float)val1.y * sigmoid<float>((float)val2.y);

        out_ptr[id] = local_out;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void mask_bias_glu(__nv_bfloat16* out,
                              const __nv_bfloat16* __restrict in,
                              const __nv_bfloat16* __restrict bias,
                              int m,
                              int n,
                              const __nv_bfloat16* __restrict attr_mask,
                              int seq_len)
{
    __nv_bfloat162* out_ptr = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;
    // const __nv_bfloat162* in_ptr = (__nv_bfloat162*)in;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        int s_id = m_id % seq_len;
        int b_id = m_id / seq_len;
        __nv_bfloat16 cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];

        // in = in + m_id * 2 * n;
        const __nv_bfloat162* in_ptr = (__nv_bfloat162*)in + m_id * 2 * n;

        __nv_bfloat162 val1 = float2type2<__nv_bfloat162>(0.f);
        __nv_bfloat162 val2 = float2type2<__nv_bfloat162>(0.f);

        if (cur_mask != __nv_bfloat16(0.f)) {
            val1 = in_ptr[n_id];
            val2 = in_ptr[n_id + n];
        }

        if (bias != nullptr) {
            __nv_bfloat162 bias1 = ldg(&bias_ptr[n_id]);
            __nv_bfloat162 bias2 = ldg(&bias_ptr[n_id + n]);
            val1 = bf16hadd2(val1, bias1);
            val2 = bf16hadd2(val2, bias2);
        }
        __nv_bfloat162 local_out;
        local_out.x = (float)val1.x * sigmoid<float>((float)val2.x);
        local_out.y = (float)val1.y * sigmoid<float>((float)val2.y);

        out_ptr[id] = local_out;
    }
}
#endif

template<typename T>
void invokeMaskBiasGlu(T* out,
                       const T* in,
                       const T* bias,
                       const int m,
                       const int n,
                       const T* attr_mask,
                       const int seq_len,
                       cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = ceil(m * n / 1024.);
    }
    mask_bias_glu<T><<<grid, block, 0, stream>>>(out, in, bias, m, n / data_type_factor, attr_mask, seq_len);
}

template void invokeMaskBiasGlu(float* out,
                                const float* in,
                                const float* bias,
                                const int m,
                                const int n,
                                const float* attr_mask,
                                const int seq_len,
                                cudaStream_t stream);
template void invokeMaskBiasGlu(half* out,
                                const half* in,
                                const half* bias,
                                const int m,
                                const int n,
                                const half* attr_mask,
                                const int seq_len,
                                cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeMaskBiasGlu(__nv_bfloat16* out,
                                const __nv_bfloat16* in,
                                const __nv_bfloat16* bias,
                                const int m,
                                const int n,
                                const __nv_bfloat16* attr_mask,
                                const int seq_len,
                                cudaStream_t stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void bias_glu(T* out, const T* __restrict in, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        // in = in + m_id * 2 * n;
        const T* in_ptr = in + m_id * 2 * n;

        float val1 = in_ptr[n_id];
        float val2 = in_ptr[n_id + n];

        if (bias != nullptr) {
            val1 = val1 + ldg(&bias[n_id]);
            val2 = val2 + ldg(&bias[n_id + n]);
        }
        out[id] = val1 * sigmoid<float>(val2);
    }
}

template<>
__global__ void bias_glu(half* out, const half* __restrict in, const half* __restrict bias, int m, int n)
{
    half2* out_ptr = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        // const half* attr_mask = attr_mask + ;

        const half2* in_ptr = (half2*)in + m_id * 2 * n;
        // in = in + m_id * 2 * n;

        half2 val1 = in_ptr[n_id];
        half2 val2 = in_ptr[n_id + n];

        if (bias != nullptr) {
            half2 bias1 = __ldg(&bias_ptr[n_id]);
            half2 bias2 = __ldg(&bias_ptr[n_id + n]);
            val1 = hadd2(val1, bias1);
            val2 = hadd2(val2, bias2);
        }
        half2 local_out;
        local_out.x = (float)val1.x * sigmoid<float>((float)val2.x);
        local_out.y = (float)val1.y * sigmoid<float>((float)val2.y);

        out_ptr[id] = local_out;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void
bias_glu(__nv_bfloat16* out, const __nv_bfloat16* __restrict in, const __nv_bfloat16* __restrict bias, int m, int n)
{
    __nv_bfloat162* out_ptr = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;
    // const __nv_bfloat162* in_ptr = (__nv_bfloat162*)in;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;

        // in = in + m_id * 2 * n;
        const __nv_bfloat162* in_ptr = (__nv_bfloat162*)in + m_id * 2 * n;

        __nv_bfloat162 val1 = in_ptr[n_id];
        __nv_bfloat162 val2 = in_ptr[n_id + n];

        if (bias != nullptr) {
            __nv_bfloat162 bias1 = ldg(&bias_ptr[n_id]);
            __nv_bfloat162 bias2 = ldg(&bias_ptr[n_id + n]);
            val1 = bf16hadd2(val1, bias1);
            val2 = bf16hadd2(val2, bias2);
        }
        __nv_bfloat162 local_out;
        local_out.x = (float)val1.x * sigmoid<float>((float)val2.x);
        local_out.y = (float)val1.y * sigmoid<float>((float)val2.y);

        out_ptr[id] = local_out;
    }
}
#endif

template<typename T>
void invokeBiasGlu(T* out, const T* in, const T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = ceil(m * n / 1024.);
    }
    bias_glu<T><<<grid, block, 0, stream>>>(out, in, bias, m, n / data_type_factor);
}

template void
invokeBiasGlu(float* out, const float* in, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeBiasGlu(half* out, const half* in, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBiasGlu(__nv_bfloat16* out,
                            const __nv_bfloat16* in,
                            const __nv_bfloat16* bias,
                            const int m,
                            const int n,
                            cudaStream_t stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void mask_bias(
    T* out, const T* __restrict in, const T* __restrict bias, int m, int n, const T* __restrict attr_mask, int seq_len)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        int s_id = m_id % seq_len;
        int b_id = m_id / seq_len;

        T cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];
        const T* in_ptr = in + m_id * n;

        float val1 = 0.f;

        if (cur_mask != T(0.f)) {
            val1 = in[n_id];

            if (bias != nullptr) {
                val1 = val1 + ldg(&bias[n_id]);
            }
        }
        out[id] = val1;
    }
}

template<>
__global__ void mask_bias(half* out,
                          const half* __restrict in,
                          const half* __restrict bias,
                          int m,
                          int n,
                          const half* __restrict attr_mask,
                          int seq_len)
{
    half2* out_ptr = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        int s_id = m_id % seq_len;
        int b_id = m_id / seq_len;
        half cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];
        const half2* in_ptr = (const half2*)in + m_id * n;

        half2 val1 = float2type2<half2>(0.f);

        if (cur_mask != half(0.f)) {
            val1 = in_ptr[n_id];

            if (bias != nullptr) {
                half2 bias1 = __ldg(&bias_ptr[n_id]);
                val1 = hadd2(val1, bias1);
            }
        }
        out_ptr[id] = val1;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void mask_bias(__nv_bfloat16* out,
                          const __nv_bfloat16* __restrict in,
                          const __nv_bfloat16* __restrict bias,
                          int m,
                          int n,
                          const __nv_bfloat16* __restrict attr_mask,
                          int seq_len)
{
    __nv_bfloat162* out_ptr = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int m_id = id / n;
        int n_id = id % n;
        int s_id = m_id % seq_len;
        int b_id = m_id / seq_len;
        __nv_bfloat16 cur_mask = attr_mask[b_id * seq_len * seq_len + s_id];
        const __nv_bfloat162* in_ptr = (const __nv_bfloat162*)in + m_id * n;

        __nv_bfloat162 val1 = float2type2<__nv_bfloat162>(0.f);

        if (cur_mask != __nv_bfloat16(0.f)) {
            val1 = in_ptr[n_id];

            if (bias != nullptr) {
                __nv_bfloat162 bias1 = ldg(&bias_ptr[n_id]);
                val1 = bf16hadd2(val1, bias1);
            }
        }
        out_ptr[id] = val1;
    }
}
#endif

template<typename T>
void invokeMaskBias(T* out,
                    const T* in,
                    const T* bias,
                    const int m,
                    const int n,
                    const T* attr_mask,
                    const int seq_len,
                    cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = ceil(m * n / 1024.);
    }
    mask_bias<T><<<grid, block, 0, stream>>>(out, in, bias, m, n / data_type_factor, attr_mask, seq_len);
}

template void invokeMaskBias(float* out,
                             const float* in,
                             const float* bias,
                             const int m,
                             const int n,
                             const float* attr_mask,
                             const int seq_len,
                             cudaStream_t stream);
template void invokeMaskBias(half* out,
                             const half* in,
                             const half* bias,
                             const int m,
                             const int n,
                             const half* attr_mask,
                             const int seq_len,
                             cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeMaskBias(__nv_bfloat16* out,
                             const __nv_bfloat16* in,
                             const __nv_bfloat16* bias,
                             const int m,
                             const int n,
                             const __nv_bfloat16* attr_mask,
                             const int seq_len,
                             cudaStream_t stream);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalScaleAddBiasResidualLayerNormOpt(T* normed_output,
                                                        T* output,
                                                        const T* __restrict bias,
                                                        const T* __restrict residual,
                                                        const T* __restrict gamma,
                                                        const T* __restrict beta,
                                                        int m,
                                                        int n,
                                                        float scale_input,
                                                        float scale_residual)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    T local_sum = float2type2<T>(0.0f);
    T local_scale_input = float2type2<T>(scale_input);
    T local_scale_residual = float2type2<T>(scale_residual);
#pragma unroll
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = float2type2<T>(0.0f);

        if (IS_OUTPUT) {
            val = hadd2(val, output[index]);
        }

        if (IS_BIAS) {
            val = hadd2(val, ldg(&bias[i]));
        }
        val = hmul2(val, local_scale_input);
        if (IS_RESIDUAL) {
            val = hadd2(val, hmul2(ldg(&residual[index]), local_scale_residual));
        }

        output[index] = val;

        local_sum = hadd2(local_sum, val);
    }

    mean = blockReduceSum((float)(local_sum.x + local_sum.y));

    if (threadIdx.x == 0) {
        s_mean = mean / n / 2;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val = output[blockIdx.x * n + i];
        float diff_1 = (float)(val.x) - s_mean;
        float diff_2 = (float)(val.y) - s_mean;
        local_var_sum += (diff_1 * diff_1 + diff_2 * diff_2);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n / 2 + EPS);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
    }
}

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalScaleAddBiasResidualLayerNormOpt2(T* normed_output,
                                                         T* output,
                                                         const T* __restrict bias,
                                                         const T* __restrict residual,
                                                         const T* __restrict gamma,
                                                         const T* __restrict beta,
                                                         int m,
                                                         int n,
                                                         float scale_input,
                                                         float scale_residual)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float x_sum = 0.0f;
    float x2_sum = 0.0f;
    const int b_offset = blockIdx.x * n;
    using T1 = typename TypeConverter<T>::Type;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        float val_1 = 0.0f;
        float val_2 = 0.0f;
        T tmp;

        if (IS_OUTPUT) {
            tmp = ldg(&output[index]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }

        if (IS_BIAS) {
            tmp = ldg(&bias[i]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        val_1 *= scale_input;
        val_2 *= scale_input;

        if (IS_RESIDUAL) {
            tmp = ldg(&residual[index]);
            val_1 += static_cast<float>(tmp.x) * scale_residual;
            val_2 += static_cast<float>(tmp.y) * scale_residual;
        }

        tmp.x = float2type<T1>(val_1);
        tmp.y = float2type<T1>(val_2);
        output[index] = tmp;
        x_sum += val_1 + val_2;
        x2_sum += val_1 * val_1 + val_2 * val_2;
    }
    float sums[2];
    sums[0] = x_sum;
    sums[1] = x2_sum;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean = sums[0] / n / 2;
        s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + EPS);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
    }
}

#define HALF_LAYERNORM_OPT(UNROLL_FACTOR)                                                                              \
    if (bias != nullptr)                                                                                               \
        generalScaleAddBiasResidualLayerNormOpt<T2, true, true, true, true, UNROLL_FACTOR>                             \
            <<<grid, block, 0, stream>>>((T2*)norm_output,                                                             \
                                         (T2*)output,                                                                  \
                                         (const T2*)bias,                                                              \
                                         (const T2*)input,                                                             \
                                         (const T2*)gamma,                                                             \
                                         (const T2*)beta,                                                              \
                                         m,                                                                            \
                                         half_n,                                                                       \
                                         scale_input,                                                                  \
                                         scale_residual);                                                              \
    else                                                                                                               \
        generalScaleAddBiasResidualLayerNormOpt<T2, true, false, true, true, UNROLL_FACTOR>                            \
            <<<grid, block, 0, stream>>>((T2*)norm_output,                                                             \
                                         (T2*)output,                                                                  \
                                         (const T2*)bias,                                                              \
                                         (const T2*)input,                                                             \
                                         (const T2*)gamma,                                                             \
                                         (const T2*)beta,                                                              \
                                         m,                                                                            \
                                         half_n,                                                                       \
                                         scale_input,                                                                  \
                                         scale_residual);

#define HALF_LAYERNORM_OPT2(UNROLL_FACTOR)                                                                             \
    if (bias != nullptr)                                                                                               \
        generalScaleAddBiasResidualLayerNormOpt2<T2, true, true, true, true, UNROLL_FACTOR>                            \
            <<<grid, block, 0, stream>>>((T2*)norm_output,                                                             \
                                         (T2*)output,                                                                  \
                                         (const T2*)bias,                                                              \
                                         (const T2*)input,                                                             \
                                         (const T2*)gamma,                                                             \
                                         (const T2*)beta,                                                              \
                                         m,                                                                            \
                                         half_n,                                                                       \
                                         scale_input,                                                                  \
                                         scale_residual);                                                              \
    else                                                                                                               \
        generalScaleAddBiasResidualLayerNormOpt2<T2, true, false, true, true, UNROLL_FACTOR>                           \
            <<<grid, block, 0, stream>>>((T2*)norm_output,                                                             \
                                         (T2*)output,                                                                  \
                                         (const T2*)bias,                                                              \
                                         (const T2*)input,                                                             \
                                         (const T2*)gamma,                                                             \
                                         (const T2*)beta,                                                              \
                                         m,                                                                            \
                                         half_n,                                                                       \
                                         scale_input,                                                                  \
                                         scale_residual);

template<typename T>
__global__ void generalScaleAddBiasResidualLayerNorm(const T* __restrict input,
                                                     const T* __restrict gamma,
                                                     const T* __restrict beta,
                                                     const T* __restrict bias,
                                                     T* output,
                                                     T* norm_output,
                                                     int m,
                                                     int n,
                                                     float scale_input,
                                                     float scale_residual)

{
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float local_out = (float)(output[blockIdx.x * n + i]);
        if (bias != nullptr) {
            local_out += (float)(ldg(&bias[i]));
        }
        local_out *= scale_input;
        local_out += (float)(ldg(&input[blockIdx.x * n + i])) * scale_residual;

        output[blockIdx.x * n + i] = (T)local_out;
        local_sum += local_out;
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(output[blockIdx.x * n + i]) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + EPS);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)(ldg(&beta[i]));
        if (norm_output != nullptr)
            norm_output[blockIdx.x * n + i] =
                (T)((((float)output[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);
    }
}

template<typename T>
void invokeGeneralScaleAddBiasResidualPreLayerNorm(T* output,
                                                   T* norm_output,
                                                   const T* input,
                                                   const T* gamma,
                                                   const T* beta,
                                                   const T* bias,
                                                   int m,
                                                   int n,
                                                   cudaStream_t stream,
                                                   int opt_version,
                                                   float scale_input,
                                                   float scale_residual)

{
    if (opt_version > 0 && sizeof(T) == 2 && n % 2 == 0) {
        dim3 grid(m);
        int half_n = n / 2;
        int half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int rolls_per_thread = half_n / block.x;
        int unroll_factor = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;
        if (opt_version == 1) {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT(8);
            }
        }
        else {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT2(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT2(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT2(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT2(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT2(8);
            }
        }
    }
    else {

        dim3 grid(m);
        dim3 block(min(n, 1024));

        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
        */

        if (n % 32 != 0) {
            block.x = 1024;
        }

        block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

        /* should pay attention to the rsqrt precision*/
        generalScaleAddBiasResidualLayerNorm<T><<<grid, block, 0, stream>>>(
            input, gamma, beta, bias, output, norm_output, m, n, scale_input, scale_residual);  // For gpt-3
    }
}

#undef HALF_LAYERNORM_OPT
#undef HALF_LAYERNORM_OPT2

template void invokeGeneralScaleAddBiasResidualPreLayerNorm(float* output,
                                                            float* norm_output,
                                                            const float* input,
                                                            const float* gamma,
                                                            const float* beta,
                                                            const float* bias,
                                                            int m,
                                                            int n,
                                                            cudaStream_t stream,
                                                            int opt_version,
                                                            float scale_input,
                                                            float scale_residual);

template void invokeGeneralScaleAddBiasResidualPreLayerNorm(half* output,
                                                            half* norm_output,
                                                            const half* input,
                                                            const half* gamma,
                                                            const half* beta,
                                                            const half* bias,
                                                            int m,
                                                            int n,
                                                            cudaStream_t stream,
                                                            int opt_version,
                                                            float scale_input,
                                                            float scale_residual);

#ifdef ENABLE_BF16
template void invokeGeneralScaleAddBiasResidualPreLayerNorm(__nv_bfloat16* output,
                                                            __nv_bfloat16* norm_output,
                                                            const __nv_bfloat16* input,
                                                            const __nv_bfloat16* gamma,
                                                            const __nv_bfloat16* beta,
                                                            const __nv_bfloat16* bias,
                                                            int m,
                                                            int n,
                                                            cudaStream_t stream,
                                                            int opt_version,
                                                            float scale_input,
                                                            float scale_residual);

#endif

//////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void scaleAddBiasResidual(
    T* output, const T* input, const T* bias, const int m, const int n, float scale_input, float scale_residual)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        float local_val = output[blockIdx.x * n + col_index];
        float bias_val = (bias == nullptr) ? (0.0f) : (float)bias[col_index];
        local_val += bias_val;
        local_val *= scale_input;
        local_val += (float)input[blockIdx.x * n + col_index] * scale_residual;
        output[blockIdx.x * n + col_index] = local_val;
    }
}

template<typename T>
void invokeScaleAddBiasResidual(T* output,
                                const T* input,
                                const T* bias,
                                const int m,
                                const int n,
                                cudaStream_t stream,
                                float scale_input,
                                float scale_residual)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    scaleAddBiasResidual<<<grid, block, 0, stream>>>(output, input, bias, m, n, scale_input, scale_residual);
}

template void invokeScaleAddBiasResidual(float* output,
                                         const float* input,
                                         const float* bias,
                                         const int m,
                                         const int n,
                                         cudaStream_t stream,
                                         float scale_input,
                                         float scale_residual);

template void invokeScaleAddBiasResidual(half* output,
                                         const half* input,
                                         const half* bias,
                                         const int m,
                                         const int n,
                                         cudaStream_t stream,
                                         float scale_input,
                                         float scale_residual);

#ifdef ENABLE_BF16
template void invokeScaleAddBiasResidual(__nv_bfloat16* output,
                                         const __nv_bfloat16* input,
                                         const __nv_bfloat16* bias,
                                         const int m,
                                         const int n,
                                         cudaStream_t stream,
                                         float scale_input,
                                         float scale_residual);
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void ConformerDepthwiseConvBiasSilu(T* out,
                                               const T* in,
                                               const T* weight,
                                               const T* bias,
                                               const int batch_size,
                                               const int seq_len,
                                               const int hidden_unit,
                                               const int kernel_size,
                                               const int pad_size)
{
    int c_id = threadIdx.x;
    int s_id = blockIdx.x % seq_len;
    int b_id = blockIdx.x / seq_len;

    int s_start = s_id - pad_size;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(pad_size - s_id, 0);

    in = in + b_id * seq_len * hidden_unit;
    weight = weight + c_id;

    float val = 0.0f;
    for (int i = s_start; i < s_end; ++i) {
        val += (float)in[i * hidden_unit + c_id] * (float)weight[(k_start + i - s_start) * hidden_unit];
    }
    val = val + (float)bias[c_id];
    val = val * sigmoid<float>(val);

    out[blockIdx.x * hidden_unit + c_id] = val;
}

template<typename T>
void invokeConformerDepthwiseConvBiasSilu(T* out,
                                          const T* in,
                                          const T* weight,
                                          const T* bias,
                                          const int batch_size,
                                          const int seq_len,
                                          const int hidden_unit,
                                          const int kernel_size,
                                          const int pad_size,
                                          cudaStream_t stream)
{
    FT_CHECK(hidden_unit <= 1024);
    ConformerDepthwiseConvBiasSilu<T><<<batch_size * seq_len, hidden_unit, 0, stream>>>(
        out, in, weight, bias, batch_size, seq_len, hidden_unit, kernel_size, pad_size);
}
template void invokeConformerDepthwiseConvBiasSilu(float* out,
                                                   const float* in,
                                                   const float* weight,
                                                   const float* bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int hidden_unit,
                                                   const int kernel_size,
                                                   const int pad_size,
                                                   cudaStream_t stream);

template void invokeConformerDepthwiseConvBiasSilu(half* out,
                                                   const half* in,
                                                   const half* weight,
                                                   const half* bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int hidden_unit,
                                                   const int kernel_size,
                                                   const int pad_size,
                                                   cudaStream_t stream);

#ifdef ENABLE_BF16

template void invokeConformerDepthwiseConvBiasSilu(__nv_bfloat16* out,
                                                   const __nv_bfloat16* in,
                                                   const __nv_bfloat16* weight,
                                                   const __nv_bfloat16* bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int hidden_unit,
                                                   const int kernel_size,
                                                   const int pad_size,
                                                   cudaStream_t stream);
#endif

/////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void VarLenConformerDepthwiseConvBiasSilu(T* out,
                                                     const T* in,
                                                     const T* weight,
                                                     const T* bias,
                                                     const int* bid_start_end,
                                                     const T* bias_before_glu,
                                                     const int m,
                                                     const int batch_size,
                                                     const int seq_len,
                                                     const int hidden_unit,
                                                     const int kernel_size,
                                                     const int pad_size)
{
    __shared__ int b_start, b_end;
    if (threadIdx.x == 0) {
        // b_id = bid_start_end[blockIdx.x * 3];
        b_start = bid_start_end[blockIdx.x * 3 + 1];
        b_end = bid_start_end[blockIdx.x * 3 + 2];
    }
    __syncthreads();

    int cur_len = b_end - b_start;

    int c_id = threadIdx.x;
    int s_id = blockIdx.x - b_start;

    int s_start = s_id - pad_size;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(pad_size - s_id, 0);

    in = in + b_start * hidden_unit;
    weight = weight + c_id;

    float val = 0.0f;
    for (int i = s_start; i < s_end; ++i) {
        float cur_in = 0.f;
        if (i < cur_len)
            cur_in = (float)in[i * hidden_unit + c_id];
        else {
            // glu(0+bias)
            cur_in =
                (float)ldg(&bias_before_glu[c_id]) * sigmoid<float>((float)ldg(&bias_before_glu[c_id + hidden_unit]));
        }
        val += cur_in * (float)weight[(k_start + i - s_start) * hidden_unit];
    }
    val = val + (float)bias[c_id];
    val = val * sigmoid<float>(val);

    out[blockIdx.x * hidden_unit + c_id] = val;
}

template<typename T>
void invokeVarLenConformerDepthwiseConvBiasSilu(T* out,
                                                const T* in,
                                                const T* weight,
                                                const T* bias,
                                                const int* bid_start_end,
                                                const T* bias_before_glu,
                                                const int m,
                                                const int batch_size,
                                                const int seq_len,
                                                const int hidden_unit,
                                                const int kernel_size,
                                                const int pad_size,
                                                cudaStream_t stream)
{
    FT_CHECK(hidden_unit <= 1024);
    VarLenConformerDepthwiseConvBiasSilu<T><<<m, hidden_unit, 0, stream>>>(out,
                                                                           in,
                                                                           weight,
                                                                           bias,
                                                                           bid_start_end,
                                                                           bias_before_glu,
                                                                           m,
                                                                           batch_size,
                                                                           seq_len,
                                                                           hidden_unit,
                                                                           kernel_size,
                                                                           pad_size);
}
template void invokeVarLenConformerDepthwiseConvBiasSilu(float* out,
                                                         const float* in,
                                                         const float* weight,
                                                         const float* bias,
                                                         const int* bid_start_end,
                                                         const float* bias_before_glu,
                                                         const int m,
                                                         const int batch_size,
                                                         const int seq_len,
                                                         const int hidden_unit,
                                                         const int kernel_size,
                                                         const int pad_size,
                                                         cudaStream_t stream);

template void invokeVarLenConformerDepthwiseConvBiasSilu(half* out,
                                                         const half* in,
                                                         const half* weight,
                                                         const half* bias,
                                                         const int* bid_start_end,
                                                         const half* bias_before_glu,
                                                         const int m,
                                                         const int batch_size,
                                                         const int seq_len,
                                                         const int hidden_unit,
                                                         const int kernel_size,
                                                         const int pad_size,
                                                         cudaStream_t stream);

#ifdef ENABLE_BF16

template void invokeVarLenConformerDepthwiseConvBiasSilu(__nv_bfloat16* out,
                                                         const __nv_bfloat16* in,
                                                         const __nv_bfloat16* weight,
                                                         const __nv_bfloat16* bias,
                                                         const int* bid_start_end,
                                                         const __nv_bfloat16* bias_before_glu,
                                                         const int m,
                                                         const int batch_size,
                                                         const int seq_len,
                                                         const int hidden_unit,
                                                         const int kernel_size,
                                                         const int pad_size,
                                                         cudaStream_t stream);
#endif

////////////////////////////////////////////////////////////////////////////

__global__ void
getBatchIDStartEndKernel(int* bid_start_end, const int* sequence_length, const int batch_size, const int seq_len)
{
    int index = 0;
    int total_seq_len = 0;
    for (int i = 0; i < batch_size; i++) {
        const int seq_len = sequence_length[i];
        for (int j = 0; j < seq_len; j++) {
            bid_start_end[index * 3] = i;
            bid_start_end[index * 3 + 1] = total_seq_len;
            bid_start_end[index * 3 + 2] = total_seq_len + seq_len;
            index++;
        }
        total_seq_len += seq_len;
    }
}

void invokeGetBatchIDStartEnd(
    int* bid_start_end, const int* sequence_length, const int batch_size, const int seq_len, cudaStream_t stream)
{
    getBatchIDStartEndKernel<<<1, 1, 0, stream>>>(bid_start_end, sequence_length, batch_size, seq_len);
}

////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void bias_rebuild_padding(const T* src, T* dst, const T* bias, const int* padding_offset, const int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int dst_seq_id = bid + padding_offset[bid];
    const int src_seq_id = bid;

    for (int i = tid; i < n; i += blockDim.x) {
        if (bias != nullptr)
            dst[dst_seq_id * n + i] = src[src_seq_id * n + i] + bias[i];
        else
            dst[dst_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template<typename T>
void invokeBiasRebuildPadding(
    T* dst, const T* src, const T* bias, const int* padding_offset, const int m, const int n, cudaStream_t stream)
{
    // src: [token_num, hidden_dim]
    // dst: [batch_size*max_seq_len, hidden_dim]
    bias_rebuild_padding<<<m, 256, 0, stream>>>(src, dst, bias, padding_offset, n);
}

template void invokeBiasRebuildPadding(float* dst,
                                       const float* src,
                                       const float* bias,
                                       const int* padding_offset,
                                       const int token_num,
                                       const int hidden_dim,
                                       cudaStream_t stream);

template void invokeBiasRebuildPadding(half* dst,
                                       const half* src,
                                       const half* bias,
                                       const int* padding_offset,
                                       const int token_num,
                                       const int hidden_dim,
                                       cudaStream_t stream);

template void invokeBiasRebuildPadding(__nv_bfloat16* dst,
                                       const __nv_bfloat16* src,
                                       const __nv_bfloat16* bias,
                                       const int* padding_offset,
                                       const int token_num,
                                       const int hidden_dim,
                                       cudaStream_t stream);

///////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void repeat_beam_size(T* out, const T* __restrict in, int beam_size, int n)
{
    int bid = blockIdx.x / beam_size;
    out = out + blockIdx.x * n;
    in = in + bid * n;
    for (int id = threadIdx.x; id < n; id += blockDim.x) {
        out[id] = in[id];
    }
}

template<typename T>
void invokeRepeatBeamSize(T* out, const T* in, const int m, const int n, const int beam_size, cudaStream_t stream)
{
    dim3 block, grid;
    block.x = std::min(1024, n);
    grid.x = m * beam_size;
    repeat_beam_size<T><<<grid, block, 0, stream>>>(out, in, beam_size, n);
}

template void
invokeRepeatBeamSize(float* out, const float* in, const int m, const int n, const int beam_size, cudaStream_t stream);
template void
invokeRepeatBeamSize(half* out, const half* in, const int m, const int n, const int beam_size, cudaStream_t stream);
template void
invokeRepeatBeamSize(int* out, const int* in, const int m, const int n, const int beam_size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeRepeatBeamSize(
    __nv_bfloat16* out, const __nv_bfloat16* in, const int m, const int n, const int beam_size, cudaStream_t stream);
#endif

////////////////////////////////////////////////////////////////////////////
__global__ void getWenetOutLensKernel(int* out, const int* in, const int batch_size, const int max_seq_len)
{
    if (threadIdx.x < batch_size) {
        int cur = in[threadIdx.x];
        int next_max = max_seq_len;

        next_max = (next_max - 1) / 2;
        cur = (cur + 1) / 2;
        cur = min(cur, next_max);

        next_max = (next_max - 1) / 2;
        cur = (cur + 1) / 2;
        cur = min(cur, next_max);

        out[threadIdx.x] = cur;
    }
}

void invokeGetWenetOutLens(int* out, const int* in, const int batch_size, const int max_seq_len, cudaStream_t stream)
{
    FT_CHECK(batch_size <= 1024);
    getWenetOutLensKernel<<<1, batch_size, 0, stream>>>(out, in, batch_size, max_seq_len);
    sync_check_cuda_error();
}
////////////////////////////////////////////////////////////////////////////
__global__ void getPaddingOffsetKernelV2(size_t* valid_word_num,
                                         int* tmp_mask_offset,
                                         const int* sequence_length,
                                         const int batch_size,
                                         const int max_seq_len)
{
    // do cumulated sum
    int total_seq_len = 0;
    int cum_offset = 0;
    int index = 0;
    for (int i = 0; i < batch_size; i++) {
        const int seq_len = sequence_length[i];
        for (int j = 0; j < seq_len; j++) {
            tmp_mask_offset[index] = cum_offset;
            index++;
        }
        cum_offset += max_seq_len - seq_len;
        total_seq_len += seq_len;
    }
    valid_word_num[0] = (size_t)total_seq_len;
}

void invokeGetPaddingOffset(size_t* d_token_num,
                            int* tmp_mask_offset,
                            const int* sequence_lengths,
                            const int batch_size,
                            const int max_seq_len,
                            cudaStream_t stream)
{
    getPaddingOffsetKernelV2<<<1, 1, 0, stream>>>(
        d_token_num, tmp_mask_offset, sequence_lengths, batch_size, max_seq_len);
    sync_check_cuda_error();
}

////////////////////////////////////////////////////////////////////////////
template<typename T, bool IS_CROSS>
__global__ void buildDecoderAttentionMaskKernel(T* attention_mask,
                                                const int* sequence_lengths1,
                                                const int max_seq_len1,
                                                const int* sequence_lengths2,
                                                const int max_seq_len2)
{
    // sequence_lengths1: [batch_size]
    // sequence_lengths2: [batch_size]
    // attention_mask: [batch_size, 1, max_seq_len1, max_seq_len2]
    const int s1_id = blockIdx.x % max_seq_len1;
    const int b_id = blockIdx.x / max_seq_len1;
    attention_mask += (b_id * max_seq_len1 + s1_id) * max_seq_len2;

    const int len1 = sequence_lengths1[b_id];
    int len2 = 0;
    if (IS_CROSS)
        len2 = sequence_lengths2[b_id];

    for (int i = threadIdx.x; i < max_seq_len2; i += blockDim.x) {
        T val = (T)(0.0f);
        if (IS_CROSS) {
            if (i < len2)
                val = (T)(1.0f);
        }
        else {
            if (i < len1 && i <= s1_id)
                val = (T)(1.0f);
        }
        attention_mask[i] = val;
    }
}

template<typename T, bool IS_CROSS>
void invokeBuildDecoderAttentionMask(T* attention_mask,
                                     const int* sequence_lengths1,
                                     const int* sequence_lengths2,
                                     const int batch_size,
                                     const int max_seq_len1,
                                     const int max_seq_len2,
                                     cudaStream_t stream)
{
    buildDecoderAttentionMaskKernel<T, IS_CROSS>
        <<<batch_size * max_seq_len1, std::min(1024, max_seq_len2), 0, stream>>>(
            attention_mask, sequence_lengths1, max_seq_len1, sequence_lengths2, max_seq_len2);
}

template void invokeBuildDecoderAttentionMask<float, false>(float* attention_mask,
                                                            const int* sequence_lengths1,
                                                            const int* sequence_lengths2,
                                                            const int batch_size,
                                                            const int max_seq_len1,
                                                            const int max_seq_len2,
                                                            cudaStream_t stream);
template void invokeBuildDecoderAttentionMask<float, true>(float* attention_mask,
                                                           const int* sequence_lengths1,
                                                           const int* sequence_lengths2,
                                                           const int batch_size,
                                                           const int max_seq_len1,
                                                           const int max_seq_len2,
                                                           cudaStream_t stream);
template void invokeBuildDecoderAttentionMask<half, false>(half* attention_mask,
                                                           const int* sequence_lengths1,
                                                           const int* sequence_lengths2,
                                                           const int batch_size,
                                                           const int max_seq_len1,
                                                           const int max_seq_len2,
                                                           cudaStream_t stream);
template void invokeBuildDecoderAttentionMask<half, true>(half* attention_mask,
                                                          const int* sequence_lengths1,
                                                          const int* sequence_lengths2,
                                                          const int batch_size,
                                                          const int max_seq_len1,
                                                          const int max_seq_len2,
                                                          cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBuildDecoderAttentionMask<__nv_bfloat16, false>(__nv_bfloat16* attention_mask,
                                                                    const int* sequence_lengths1,
                                                                    const int* sequence_lengths2,
                                                                    const int batch_size,
                                                                    const int max_seq_len1,
                                                                    const int max_seq_len2,
                                                                    cudaStream_t stream);
template void invokeBuildDecoderAttentionMask<__nv_bfloat16, true>(__nv_bfloat16* attention_mask,
                                                                   const int* sequence_lengths1,
                                                                   const int* sequence_lengths2,
                                                                   const int batch_size,
                                                                   const int max_seq_len1,
                                                                   const int max_seq_len2,
                                                                   cudaStream_t stream);

#endif
//////////////////////////////////////////////////////////////////////////////

template<typename T, int EPT>
__global__ void biasLogSoftmaxKernel(float* log_probs,
                                     const T* logits,
                                     const T* bias,
                                     const int* lengths,
                                     const size_t max_input_length,
                                     const size_t batch_size,
                                     const size_t vocab_size,
                                     const size_t vocab_size_padded)
{
    constexpr bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    int tidx = threadIdx.x;  // vocab dim
    int bidx = blockIdx.x;   // batch dim
    int step = blockIdx.y;   // step dim

    __shared__ float s_max_logit, s_sum_logit;

    bool is_valid = (bidx < batch_size) && (step < max_input_length);
    if (lengths != nullptr)
        is_valid = is_valid && (step < lengths[bidx]);
    if (is_valid) {
        // reposition logits to data for the current batch.
        logits += bidx * max_input_length * vocab_size_padded + step * vocab_size_padded;
        log_probs += bidx * max_input_length * vocab_size + step * vocab_size;

        // load and add bias
        T local_logit[EPT];
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            int cur_idx = tidx + i * blockDim.x;
            if (cur_idx < vocab_size) {
                local_logit[i] = logits[cur_idx];
                if (bias != nullptr)
                    local_logit[i] = (float)local_logit[i] + (float)bias[cur_idx];
            }
            // else
            //     local_logit[i] = -MAX_T_VAL;
        }
        // Find max(logits).
        float local_max = -MAX_T_VAL;
        float val = -MAX_T_VAL;
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            int cur_idx = tidx + i * blockDim.x;
            if (cur_idx < vocab_size) {
                val = static_cast<float>(local_logit[i]);
                local_max = fmax(local_max, val);
            }
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (tidx == 0) {
            s_max_logit = max_val;
        }
        __syncthreads();

        // Calculate the denominator: sum_i exp(logits[i])
        float local_sum_exp = 0.0f;
// val = 0.0f;
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            int cur_idx = tidx + i * blockDim.x;
            if (cur_idx < vocab_size) {
                val = __expf(static_cast<float>(local_logit[i]) - s_max_logit);
                local_sum_exp += val;
            }
        }

        float sum_exp = blockDim.x <= 32 ? warpReduceSum(local_sum_exp) : blockReduceSum<float>(local_sum_exp);
        if (tidx == 0) {
            s_sum_logit = sum_exp;
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            int cur_idx = tidx + i * blockDim.x;
            if (cur_idx < vocab_size)
                log_probs[cur_idx] = static_cast<float>(local_logit[i]) - s_max_logit - __logf(s_sum_logit + 1e-9f);
        }
    }
}

template<typename T>
void invokeBiasLogSoftmax(float* log_probs,
                          const T* logits,
                          const T* bias,
                          const int* lengths,
                          const size_t max_input_length,
                          const size_t batch_size,
                          const size_t vocab_size,
                          const size_t vocab_size_padded,
                          bool batch_first,
                          cudaStream_t stream)
{
    FT_CHECK(vocab_size <= 768 * 6);
    dim3 block, grid;
    block.x = 768;
    grid.x = batch_size;
    grid.y = max_input_length;
    biasLogSoftmaxKernel<T, 6><<<grid, block, 0, stream>>>(
        log_probs, logits, bias, lengths, max_input_length, batch_size, vocab_size, vocab_size_padded);
}

template void invokeBiasLogSoftmax<float>(float* log_probs,
                                          const float* logits,
                                          const float* bias,
                                          const int* lengths,
                                          const size_t max_input_length,
                                          const size_t batch_size,
                                          const size_t vocab_size,
                                          const size_t vocab_size_padded,
                                          bool batch_first,
                                          cudaStream_t stream);
template void invokeBiasLogSoftmax<half>(float* log_probs,
                                         const half* logits,
                                         const half* bias,
                                         const int* lengths,
                                         const size_t max_input_length,
                                         const size_t batch_size,
                                         const size_t vocab_size,
                                         const size_t vocab_size_padded,
                                         bool batch_first,
                                         cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBiasLogSoftmax<__nv_bfloat16>(float* log_probs,
                                                  const __nv_bfloat16* logits,
                                                  const __nv_bfloat16* bias,
                                                  const int* lengths,
                                                  const size_t max_input_length,
                                                  const size_t batch_size,
                                                  const size_t vocab_size,
                                                  const size_t vocab_size_padded,
                                                  bool batch_first,
                                                  cudaStream_t stream);
#endif
////////////////////////////////////////////////////////////////////////////
template<typename T>
void PrintRow(const T* data, int ri, int m, int n)
{
    if (ri < 0 || ri >= m)
        return;
    const T* cur = data + ri * n;

    int cc = 6;
    for (int i = 0; i < std::min(cc, n); ++i)
        std::cout << (float)cur[i] << ",";
    std::cout << "..."
              << ",";
    for (int i = n - cc - 1; i >= 0 && i < n; ++i)
        std::cout << (float)cur[i] << ",";
    std::cout << std::endl;
}
template<typename T>
void PrintCuda2D(cudaStream_t stream, const T* data, int m, int n)
{
    static int i = 0;
    T* cpu_data = new T[m * n];
    cudaMemcpyAsync(cpu_data, data, m * n * sizeof(T), ::cudaMemcpyDefault, stream);
    cudaStreamSynchronize(stream);
    std::cout << "########### id " << i++ << " ###(" << m << "," << n << ")########" << std::endl;

    PrintRow<T>(cpu_data, 0, m, n);
    PrintRow<T>(cpu_data, 1, m, n);

    PrintRow<T>(cpu_data, m - 2, m, n);
    PrintRow<T>(cpu_data, m - 1, m, n);

    std::cout << "########### sizeof(T) " << sizeof(T) << " ###########" << std::endl;
    delete[] cpu_data;
}

template void PrintCuda2D<float>(cudaStream_t stream, const float* data, int m, int n);
template void PrintCuda2D<half>(cudaStream_t stream, const half* data, int m, int n);
template void PrintCuda2D<int>(cudaStream_t stream, const int* data, int m, int n);

#ifdef ENABLE_BF16
template void PrintCuda2D<__nv_bfloat16>(cudaStream_t stream, const __nv_bfloat16* data, int m, int n);
#endif

}  // namespace fastertransformer
