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

#include "src/fastertransformer/models/wenet/WenetDecoder.h"
#include "src/fastertransformer/models/wenet/MultiHeadedAttentionLayer.h"
#include "src/fastertransformer/models/wenet/WenetKernels.h"

namespace fastertransformer {

template<typename T>
void WenetDecoder<T>::initialize()
{

    self_attention_layers_.resize(num_layer_);
    for (size_t i = 0; i < self_attention_layers_.size(); ++i)
        self_attention_layers_[i] = new MultiHeadedAttentionLayer<T>(max_batch_size_,
                                                                     max_seq_len_,
                                                                     head_num_,
                                                                     size_per_head_,
                                                                     qscaling_,
                                                                     stream_,
                                                                     cublas_wrapper_,
                                                                     allocator_,
                                                                     is_free_buffer_after_forward_);

    cross_attention_layers_.resize(num_layer_);
    for (size_t i = 0; i < cross_attention_layers_.size(); ++i)
        cross_attention_layers_[i] = new MultiHeadedAttentionLayer<T>(max_batch_size_,
                                                                      max_seq_len_,
                                                                      head_num_,
                                                                      size_per_head_,
                                                                      qscaling_,
                                                                      stream_,
                                                                      cublas_wrapper_,
                                                                      allocator_,
                                                                      is_free_buffer_after_forward_);

    ffn_layer_ = new ReluFfnLayer<T>(max_batch_size_,
                                     max_seq_len_,
                                     head_num_,
                                     size_per_head_,
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_);
}

template<typename T>
void WenetDecoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        size_t feature_size = max_batch_size_ * max_seq_len_ * hidden_units_ * sizeof(T);

        encoder_output_repeated_ = reinterpret_cast<T*>(allocator_->malloc(feature_size, false));
        encoder_sequence_length_repeated_ =
            reinterpret_cast<int*>(allocator_->malloc(max_batch_size_ * sizeof(int), false));
        self_attn_mask_ =
            reinterpret_cast<T*>(allocator_->malloc(max_batch_size_ * max_seq_len_ * max_seq_len_ * sizeof(T), false));
        cross_attn_mask_ =
            reinterpret_cast<T*>(allocator_->malloc(max_batch_size_ * max_seq_len_ * max_seq_len_ * sizeof(T), false));

        decoder_normed_input_ = reinterpret_cast<T*>(allocator_->malloc(feature_size, false));
        self_attn_output_ = reinterpret_cast<T*>(allocator_->malloc(feature_size, false));
        normed_self_attn_output_ = reinterpret_cast<T*>(allocator_->malloc(feature_size, false));
        cross_attn_output_ = reinterpret_cast<T*>(allocator_->malloc(feature_size, false));
        normed_cross_attn_output_ = reinterpret_cast<T*>(allocator_->malloc(feature_size, false));
        decoder_layer_output_ = reinterpret_cast<T*>(allocator_->malloc(feature_size, false));

        log_probs_buf_ = reinterpret_cast<float*>(
            allocator_->malloc(max_batch_size_ * max_seq_len_ * vocab_size_ * sizeof(float), false));

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void WenetDecoder<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    size_t feature_size = batch_size * seq_len * hidden_units_ * sizeof(T);

    encoder_output_repeated_ =
        reinterpret_cast<T*>(allocator_->reMalloc(encoder_output_repeated_, feature_size, false));
    encoder_sequence_length_repeated_ = reinterpret_cast<int*>(
        allocator_->reMalloc(encoder_sequence_length_repeated_, batch_size * sizeof(int), false));
    self_attn_mask_ =
        reinterpret_cast<T*>(allocator_->reMalloc(self_attn_mask_, batch_size * seq_len * seq_len * sizeof(T), false));
    cross_attn_mask_ =
        reinterpret_cast<T*>(allocator_->reMalloc(cross_attn_mask_, batch_size * seq_len * seq_len * sizeof(T), false));

    decoder_normed_input_ = (T*)allocator_->reMalloc(decoder_normed_input_, feature_size, false);
    self_attn_output_ = (T*)allocator_->reMalloc(self_attn_output_, feature_size, false);
    normed_self_attn_output_ = (T*)allocator_->reMalloc(normed_self_attn_output_, feature_size, false);
    cross_attn_output_ = (T*)allocator_->reMalloc(cross_attn_output_, feature_size, false);
    normed_cross_attn_output_ = (T*)allocator_->reMalloc(normed_cross_attn_output_, feature_size, false);
    decoder_layer_output_ = (T*)allocator_->reMalloc(decoder_layer_output_, feature_size, false);

    log_probs_buf_ = reinterpret_cast<float*>(
        allocator_->reMalloc(log_probs_buf_, max_batch_size_ * max_seq_len_ * vocab_size_ * sizeof(float), false));
}

template<typename T>
void WenetDecoder<T>::freeBuffer()
{
    allocator_->free(encoder_output_repeated_);
    allocator_->free(encoder_sequence_length_repeated_);
    allocator_->free(self_attn_mask_);
    allocator_->free(cross_attn_mask_);

    allocator_->free(decoder_normed_input_);
    allocator_->free(self_attn_output_);
    allocator_->free(normed_self_attn_output_);
    allocator_->free(cross_attn_output_);
    allocator_->free(normed_cross_attn_output_);
    allocator_->free(decoder_layer_output_);

    allocator_->free(log_probs_buf_);
}

template<typename T>
WenetDecoder<T>::WenetDecoder(size_t max_batch_size,
                              size_t max_seq_len,
                              size_t head_num,
                              size_t size_per_head,
                              size_t inter_size,
                              size_t num_layer,
                              size_t vocab_size,
                              float qscaling,
                              cudaStream_t stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator* allocator,
                              bool is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    qscaling_(qscaling),
    hidden_units_(head_num_ * size_per_head)
{
    initialize();
}

template<typename T>
WenetDecoder<T>::WenetDecoder(WenetDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    max_seq_len_(decoder.max_seq_len_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    num_layer_(decoder.num_layer_),
    vocab_size_(decoder.vocab_size_),
    qscaling_(decoder.qscaling_),
    hidden_units_(decoder.hidden_units_)
{
    initialize();
}

template<typename T>
WenetDecoder<T>::~WenetDecoder()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (size_t i = 0; i < self_attention_layers_.size(); ++i)
        delete self_attention_layers_[i];

    for (size_t i = 0; i < cross_attention_layers_.size(); ++i)
        delete cross_attention_layers_[i];

    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void WenetDecoder<T>::forward(std::vector<Tensor>* output_tensors,
                              const std::vector<Tensor>* input_tensors,
                              const WenetDecoderWeight<T>* decoder_weight)
{
    // input tensors:
    //      decoder_input [batch_size, max_seq_len, hidden_dimension],
    //      decoder_sequence_length [batch_size],
    //      encoder_output [mem_batch_size, mem_max_seq_len, memory_hidden_dimension],
    //      encoder_sequence_length [batch_size],

    // output tensors:
    //      decoder_output [batch_size, hidden_dimension],

    // TODO(yuqingding)
    // Done: build self_attn_mask
    // Done: build cross_attn_mask
    // cudagraph
    // Done: log_softmax
    // Done: merge self qkv
    // gemm config
    FT_CHECK(input_tensors->size() == 4);
    FT_CHECK(output_tensors->size() == 1);

    const size_t batch_size = (size_t)input_tensors->at(0).shape[0];
    const size_t seq_len1 = (size_t)input_tensors->at(0).shape[1];
    const size_t batch_size2 = (size_t)input_tensors->at(2).shape[0];
    const size_t seq_len2 = (size_t)input_tensors->at(2).shape[1];
    const size_t seq_len12 = std::max(seq_len1, seq_len2);
    allocateBuffer(batch_size, seq_len12);

    const size_t beam_size = batch_size / batch_size2;

    const size_t m = batch_size * seq_len1;
    const DataType data_type = getTensorType<T>();

    const int* decoder_sequence_length = (const int*)input_tensors->at(1).data;
    const T* encoder_output = (const T*)input_tensors->at(2).data;
    const int* encoder_sequence_length = (const int*)input_tensors->at(3).data;

    invokeRepeatBeamSize(encoder_output_repeated_,
                         encoder_output,
                         batch_size2,
                         seq_len2 * input_tensors->at(2).shape[2],
                         beam_size,
                         stream_);
    sync_check_cuda_error();

    invokeRepeatBeamSize(
        encoder_sequence_length_repeated_, encoder_sequence_length, batch_size2, 1, beam_size, stream_);
    sync_check_cuda_error();

    invokeBuildDecoderAttentionMask<T, false>(
        self_attn_mask_, decoder_sequence_length, nullptr, batch_size, seq_len1, seq_len1, stream_);
    sync_check_cuda_error();

    invokeBuildDecoderAttentionMask<T, true>(cross_attn_mask_,
                                             decoder_sequence_length,
                                             encoder_sequence_length_repeated_,
                                             batch_size,
                                             seq_len1,
                                             seq_len2,
                                             stream_);

    sync_check_cuda_error();

    for (uint l = 0; l < num_layer_; l++) {
        const T* decoder_input = (const T*)((l == 0) ? input_tensors->at(0).data : decoder_layer_output_);
        // T* decoder_output = (T*)((l == num_layer_ - 1) ? output_tensors->at(0).data : decoder_layer_output_);

        invokeGeneralLayerNorm(decoder_normed_input_,
                               decoder_input,
                               decoder_weight->decoder_layer_weights[l]->pre_layernorm_weights.gamma,
                               decoder_weight->decoder_layer_weights[l]->pre_layernorm_weights.beta,
                               m,
                               hidden_units_,
                               stream_);
        sync_check_cuda_error();

        std::vector<Tensor> self_attention_input_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, seq_len1, hidden_units_}, decoder_normed_input_},
            Tensor{MEMORY_GPU, data_type, {batch_size, seq_len1, hidden_units_}, decoder_normed_input_},
            Tensor{MEMORY_GPU, data_type, {batch_size, seq_len1, hidden_units_}, decoder_normed_input_},
            Tensor{MEMORY_GPU, data_type, {batch_size, 1, seq_len1, seq_len1}, self_attn_mask_},
            Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, nullptr}};

        std::vector<Tensor> self_attention_output_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, seq_len1, hidden_units_}, self_attn_output_}};
        self_attention_layers_[l]->forward(&self_attention_output_tensors,
                                           &self_attention_input_tensors,
                                           &decoder_weight->decoder_layer_weights[l]->self_attention_weights);

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            self_attn_output_,
            normed_self_attn_output_,
            decoder_input,
            decoder_weight->decoder_layer_weights[l]->self_attn_layernorm_weights.gamma,
            decoder_weight->decoder_layer_weights[l]->self_attn_layernorm_weights.beta,
            decoder_weight->decoder_layer_weights[l]->self_attention_weights.attention_output_weight.bias,
            m,
            hidden_units_,
            stream_,
            2,
            1.0f,
            1.0f);
        sync_check_cuda_error();

        std::vector<Tensor> cross_attention_input_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, seq_len1, hidden_units_}, normed_self_attn_output_},
            Tensor{MEMORY_GPU, data_type, {batch_size, seq_len2, hidden_units_}, encoder_output_repeated_},
            Tensor{MEMORY_GPU, data_type, {batch_size, seq_len2, hidden_units_}, encoder_output_repeated_},
            Tensor{MEMORY_GPU, data_type, {batch_size, 1, seq_len1, seq_len2}, cross_attn_mask_},
            Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, nullptr}};
        std::vector<Tensor> cross_attention_output_tensors{
            Tensor{MEMORY_GPU, data_type, {batch_size, seq_len1, hidden_units_}, cross_attn_output_}};

        cross_attention_layers_[l]->forward(&cross_attention_output_tensors,
                                            &cross_attention_input_tensors,
                                            &decoder_weight->decoder_layer_weights[l]->cross_attention_weights);

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            cross_attn_output_,
            normed_cross_attn_output_,
            self_attn_output_,
            decoder_weight->decoder_layer_weights[l]->cross_attn_layernorm_weights.gamma,
            decoder_weight->decoder_layer_weights[l]->cross_attn_layernorm_weights.beta,
            decoder_weight->decoder_layer_weights[l]->cross_attention_weights.attention_output_weight.bias,
            m,
            hidden_units_,
            stream_,
            2,
            1.0f,
            1.0f);
        sync_check_cuda_error();

        std::vector<Tensor> ffn_input_tensors{
            Tensor{MEMORY_GPU, data_type, {m, hidden_units_}, normed_cross_attn_output_}};
        std::vector<Tensor> ffn_output_tensors{
            Tensor{MEMORY_GPU, data_type, {m, hidden_units_}, decoder_layer_output_}};
        ffn_layer_->forward(
            &ffn_output_tensors, &ffn_input_tensors, &decoder_weight->decoder_layer_weights[l]->ffn_weights);

        invokeScaleAddBiasResidual(decoder_layer_output_,
                                   cross_attn_output_,
                                   decoder_weight->decoder_layer_weights[l]->ffn_weights.output_weight.bias,
                                   m,
                                   hidden_units_,
                                   stream_,
                                   1.0f,
                                   1.0f);
        sync_check_cuda_error();
    }
    T* decoder_output = (T*)output_tensors->at(0).data;
    invokeGeneralLayerNorm(decoder_output,
                           decoder_layer_output_,
                           decoder_weight->after_norm_weights.gamma,
                           decoder_weight->after_norm_weights.beta,
                           m,
                           hidden_units_,
                           stream_);
    sync_check_cuda_error();

    if (true) {
        int n = vocab_size_;
        // int m = m;
        int k = hidden_units_;

        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              decoder_weight->output_layer_weights.kernel,
                              n,
                              decoder_output,
                              k,
                              (T*)log_probs_buf_,
                              n);

        float* decoder_output_ptr = (float*)output_tensors->at(0).data;

        // add bias and log_softmax
        invokeBiasLogSoftmax<T>(decoder_output_ptr,
                                (T*)log_probs_buf_,
                                decoder_weight->output_layer_weights.bias,
                                nullptr,
                                seq_len1,
                                batch_size,
                                vocab_size_,
                                vocab_size_,
                                true,
                                stream_);
        sync_check_cuda_error();
        // DEBUG_FP32(decoder_output_ptr, m, vocab_size_, "/target/tmp/dec_ls_out.bin");
        // exit(0);
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class WenetDecoder<float>;
template class WenetDecoder<half>;

}  // namespace fastertransformer