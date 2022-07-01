/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/wenet/SiluFfnLayer.h"

namespace fastertransformer {

template<typename T>
SiluFfnLayer<T>::SiluFfnLayer(size_t max_batch_size,
                              size_t max_seq_len,
                              size_t head_num,
                              size_t size_per_head,
                              size_t inter_size,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward,
                              bool sparse):
    FfnLayer<T>(max_batch_size,
                max_seq_len,
                head_num,
                size_per_head,
                inter_size,
                stream,
                cublas_wrapper,
                allocator,
                is_free_buffer_after_forward,
                sparse)
{
}

template<typename T>
SiluFfnLayer<T>::SiluFfnLayer(SiluFfnLayer<T> const& ffn_layer): FfnLayer<T>(ffn_layer)
{
}

template<typename T>
void SiluFfnLayer<T>::invokeAddBiasActivation(const int m, const T* bias)
{
    invokeAddBiasSilu<T>(inter_buf_, bias, m, inter_size_, stream_);
}

template class SiluFfnLayer<float>;
template class SiluFfnLayer<half>;
#ifdef ENABLE_BF16
template class SiluFfnLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
