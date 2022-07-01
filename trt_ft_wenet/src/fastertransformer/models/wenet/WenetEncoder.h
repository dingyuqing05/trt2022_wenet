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

#pragma once

#include <unordered_map>
#include <vector>

#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/models/wenet/ConformerConvLayer.h"
#include "src/fastertransformer/models/wenet/RelPositionAttentionLayer.h"
#include "src/fastertransformer/models/wenet/SiluFfnLayer.h"
#include "src/fastertransformer/models/wenet/WenetEncoderWeight.h"
#include "src/fastertransformer/models/wenet/FTCudaGraph.h"

namespace fastertransformer {

template<typename T>
class WenetEncoder: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;

    // meta data
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t d_model_;
    const size_t hidden_units_;
    const size_t num_layer_;

    int sm_;
    float q_scaling_;
    size_t int8_mode_;
    AttentionType attention_type_;
    bool sparse_;

    FfnLayer<T>* ffn_layer_;
    std::vector<RelPositionAttentionLayer<T>*> attention_layers_;
    ConformerConvLayer<T>* conformer_conv_layer_;

    bool is_allocate_buffer_ = false;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);
    void freeBuffer() override;
    void initialize();

    const ActivationType activation_type_;

    const size_t vocab_size_;
    const size_t beam_width_;

    // for varlen
    size_t* h_var_token_num_;
    cudaEvent_t stream_finished_;
    cudaEvent_t stream2_finished_;
    cudaStream_t stream2_;

    // for pos_emb cache and cuda graph
    bool is_enqueue_init_ = false;

    // for cuda graph
    bool use_cuda_graph_ = true;
    std::unordered_map<std::string, FTCudaGraph*> cuda_graph_pool_;
    //size_t enqueue_count_ = 0;
protected:
    T* attention_mask_ = nullptr;
    T* pos_emb_repeated_[32];
    T* pos_emb_cache_[32];

    size_t* token_num_ = nullptr;
    int* padding_offset_ = nullptr;
    int* bid_start_end_ = nullptr;

    T* normed_from_tensor_ = nullptr;

    T* ffn_out_buf_ = nullptr;
    T* normed_ffn_out_buf_ = nullptr;

    T* attn_out_buf_ = nullptr;
    T* normed_attn_out_buf_ = nullptr;

    T* conv_out_buf_ = nullptr;
    T* normed_conv_out_buf_ = nullptr;

    T* ffn2_out_buf_ = nullptr;

    T* ctc_lo_out_buf_ = nullptr;

    T* log_softmax_out_buf_ = nullptr; // unused

    size_t topk_workspace_size_ = 0;
    void* topk_workspace_ = nullptr;
public:
    WenetEncoder(size_t max_batch_size,
                 size_t max_seq_len,
                 size_t head_num,
                 size_t size_per_head,
                 size_t inter_size,
                 size_t d_model,
                 size_t num_layer,
                 size_t vocab_size,
                 size_t beam_width,
                 int sm,
                 float q_scaling,
                 size_t int8_mode,
                 cudaStream_t stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator* allocator,
                 bool is_free_buffer_after_forward,
                 AttentionType attention_type,
                 bool sparse,
                 ActivationType activation_type);

    WenetEncoder(WenetEncoder<T> const& layer);

    ~WenetEncoder();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const WenetEncoderWeight<T>* weights);

    void forward(std::unordered_map<std::string, Tensor>* output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const WenetEncoderWeight<T>* weights);

    void setStream(cudaStream_t stream) override;
};

}  // namespace fastertransformer
