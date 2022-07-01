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

#include "src/fastertransformer/models/wenet/ConformerConvLayer.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/models/wenet/WenetKernels.h"

namespace fastertransformer {

#ifdef ENABLE_BF16
/*
template void invokeRemovePadding(__nv_bfloat16* dst,
                                  const __nv_bfloat16* src,
                                  const int* padding_offset,
                                  const int token_num,
                                  const int hidden_dim,
                                  cudaStream_t stream);

template void invokeRebuildPadding(__nv_bfloat16* dst,
                                   const __nv_bfloat16* src,
                                   const int* padding_offset,
                                   const int token_num,
                                   const int hidden_dim,
                                   cudaStream_t stream);
*/
#endif
namespace {
// ugly implementation
template<typename T>
struct PaddingType {
    using Type = T;
};
template<>
struct PaddingType<__nv_bfloat16> {
    using Type = half;
};
}  // namespace

template<typename T>
void ConformerConvLayer<T>::initialize()
{
    allocateBuffer();
}

template<typename T>
void ConformerConvLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                    const std::vector<fastertransformer::Tensor>* input_tensors,
                                    const ConformerConvWeight<T>* conformer_conv_weights)
{
    // input tensors:
    //      input [batch_size, seq_len, hidden_dimension],
    //      attention_mask (batch_size, 1, seqlen, seqlen),
    //      padding_offset (h_var_token_num),
    //      bid_start_end  (h_var_token_num * 3)

    // output tensors:
    //      output [batch_size, seq_len hidden_dimension],

    bool use_varlen = false;

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 4);
    FT_CHECK(output_tensors->size() == 1);
    // FT_CHECK(isValidTokenNum(input_tensors->at(0).shape[0]));
    const int batch_size = input_tensors->at(0).shape[0];
    const int seq_len = input_tensors->at(0).shape[1];
    int m = batch_size * seq_len;
    allocateBuffer(m);

    const T* input_tensor = (const T*)input_tensors->at(0).data;
    const T* attr_mask_data = (const T*)input_tensors->at(1).data;
    T* output_tensor = (T*)output_tensors->at(0).data;

    const int* padding_offset = (const int*)input_tensors->at(2).data;
    const int* bid_start_end = (const int*)input_tensors->at(3).data;
    if (use_varlen) {
        m = input_tensors->at(2).shape[0];
        invokeRemovePadding((typename PaddingType<T>::Type*)input_remove_padding_,
                            (const typename PaddingType<T>::Type*)input_tensors->at(0).data,
                            padding_offset,
                            m,
                            head_num_ * size_per_head_,
                            stream_);
        sync_check_cuda_error();
        input_tensor = input_remove_padding_;
    }

#ifdef SPARSITY_ENABLED
    int m_tmp = m;
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    const int m_padded = m_tmp;
    if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_ * 2, m, hidden_units_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                hidden_units_ * 2,
                                m_padded,
                                hidden_units_,
                                conformer_conv_weights->pointwise_conv1_weight.sp_kernel,
                                input_tensor,
                                inter2_buf_);
    }
    else {
#endif

        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              hidden_units_ * 2,
                              m,
                              hidden_units_,
                              conformer_conv_weights->pointwise_conv1_weight.kernel,
                              hidden_units_ * 2,
                              input_tensor,
                              hidden_units_,
                              inter2_buf_,
                              hidden_units_ * 2);

#ifdef SPARSITY_ENABLED
    }
#endif
    // inter2_buf_ -> inter_buf_
    if (use_varlen) {
        invokeBiasGlu(inter_buf_,  // input_remove_padding_,
                      inter2_buf_,
                      conformer_conv_weights->pointwise_conv1_weight.bias,
                      m,
                      hidden_units_,
                      stream_);
        sync_check_cuda_error();
    }
    else {
        invokeMaskBiasGlu(inter_buf_,
                          inter2_buf_,
                          conformer_conv_weights->pointwise_conv1_weight.bias,
                          m,
                          hidden_units_,
                          attr_mask_data,
                          seq_len,
                          stream_);
    }
    sync_check_cuda_error();

    // todo: dpconv
    // inter_buf_ -> inter2_buf_
    if (use_varlen) {
        invokeVarLenConformerDepthwiseConvBiasSilu(inter2_buf_,
                                                   inter_buf_,
                                                   conformer_conv_weights->depthwise_conv_weight.kernel,
                                                   conformer_conv_weights->depthwise_conv_weight.bias,
                                                   bid_start_end,
                                                   conformer_conv_weights->pointwise_conv1_weight.bias,
                                                   m,
                                                   batch_size,
                                                   seq_len,
                                                   hidden_units_,
                                                   15,
                                                   7,
                                                   stream_);
    }
    else {
        invokeConformerDepthwiseConvBiasSilu(inter2_buf_,
                                             inter_buf_,
                                             conformer_conv_weights->depthwise_conv_weight.kernel,
                                             conformer_conv_weights->depthwise_conv_weight.bias,
                                             batch_size,
                                             seq_len,
                                             hidden_units_,
                                             15,
                                             7,
                                             stream_);
    }
    // inter2_buf -> inter_buf_
    // invokeAddBiasActivation(m, conformer_conv_weights->depthwise_conv_weight.bias);

    // inter_buf2_ -> inter_buf_
#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m, hidden_units_)) {
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                hidden_units_,
                                m_padded,
                                hidden_units_,
                                conformer_conv_weights->pointwise_conv2_weight.sp_kernel,
                                inter2_buf_,
                                inter_buf_);
    }
    else {
#endif
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              hidden_units_,
                              m,
                              hidden_units_,
                              conformer_conv_weights->pointwise_conv2_weight.kernel,
                              hidden_units_,
                              inter2_buf_,
                              hidden_units_,
                              inter_buf_,
                              hidden_units_);

        if (use_varlen) {
            cudaMemsetAsync(output_tensor, 0, batch_size * seq_len * hidden_units_ * sizeof(T), stream_);

            invokeBiasRebuildPadding(output_tensor,
                                     inter_buf_,
                                     conformer_conv_weights->pointwise_conv2_weight.bias,
                                     padding_offset,
                                     m,
                                     hidden_units_,
                                     stream_);

            m = batch_size * seq_len;
        }
        else {
            invokeMaskBias(output_tensor,
                           inter_buf_,
                           conformer_conv_weights->pointwise_conv2_weight.bias,
                           m,
                           hidden_units_,
                           attr_mask_data,
                           seq_len,
                           stream_);
        }
#ifdef SPARSITY_ENABLED
    }
#endif
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
ConformerConvLayer<T>::ConformerConvLayer(size_t max_batch_size,
                                          size_t max_seq_len,
                                          size_t head_num,
                                          size_t size_per_head,
                                          cudaStream_t stream,
                                          cublasMMWrapper* cublas_wrapper,
                                          IAllocator* allocator,
                                          bool is_free_buffer_after_forward,
                                          bool sparse,
                                          int int8_mode):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    max_token_num_(max_batch_size * max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    int8_mode_(int8_mode)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize();
}

template<typename T>
ConformerConvLayer<T>::ConformerConvLayer(ConformerConvLayer<T> const& conformer_conv_layer):
    BaseLayer(conformer_conv_layer.stream_,
              conformer_conv_layer.cublas_wrapper_,
              conformer_conv_layer.allocator_,
              conformer_conv_layer.is_free_buffer_after_forward_,
              conformer_conv_layer.cuda_device_prop_,
              conformer_conv_layer.sparse_),
    max_token_num_(conformer_conv_layer.max_token_num_),
    head_num_(conformer_conv_layer.head_num_),
    size_per_head_(conformer_conv_layer.size_per_head_),
    hidden_units_(conformer_conv_layer.hidden_units_),
    int8_mode_(conformer_conv_layer.int8_mode_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize();
}

template<typename T>
ConformerConvLayer<T>::~ConformerConvLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void ConformerConvLayer<T>::allocateBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_ == false) {
        input_remove_padding_ = (T*)allocator_->malloc(sizeof(T) * max_token_num_ * hidden_units_, false);
        inter_buf_ = (T*)allocator_->malloc(sizeof(T) * max_token_num_ * hidden_units_, false);
        inter2_buf_ = (T*)allocator_->malloc(sizeof(T) * max_token_num_ * hidden_units_ * 2, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void ConformerConvLayer<T>::allocateBuffer(size_t token_num)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(token_num <= max_token_num_);
    return; // for cuda graph

    input_remove_padding_ =
        (T*)allocator_->reMalloc(input_remove_padding_, sizeof(T) * token_num * hidden_units_, false);
    inter_buf_ = (T*)allocator_->reMalloc(inter_buf_, sizeof(T) * token_num * hidden_units_, false);
    inter2_buf_ = (T*)allocator_->reMalloc(inter2_buf_, sizeof(T) * token_num * hidden_units_ * 2, false);
    is_allocate_buffer_ = true;
}

template<typename T>
void ConformerConvLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free(input_remove_padding_);
        allocator_->free(inter_buf_);
        allocator_->free(inter2_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ConformerConvLayer<T>::isValidTokenNum(size_t token_num)
{
    if (max_token_num_ < token_num) {
        max_token_num_ = token_num;
    }
    return true;
}

template class ConformerConvLayer<float>;
template class ConformerConvLayer<half>;
#ifdef ENABLE_BF16
template class ConformerConvLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
