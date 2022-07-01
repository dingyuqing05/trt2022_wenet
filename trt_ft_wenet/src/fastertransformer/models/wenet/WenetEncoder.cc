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

#include "src/fastertransformer/models/wenet/WenetEncoder.h"

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/beam_search_topk_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"

namespace fastertransformer {

template<typename T>
void WenetEncoder<T>::initialize()
{
    check_cuda_error(cudaStreamCreate(&stream2_));
    check_cuda_error(cudaEventCreate(&stream_finished_));
    check_cuda_error(cudaEventCreate(&stream2_finished_));
    check_cuda_error(cudaMallocHost((void**)&h_var_token_num_, sizeof(size_t)));
    if(int8_mode_!=0)
    {
        FT_CHECK(activation_type_ == ActivationType::Silu);
    }
    attention_layers_.resize(num_layer_);
    for (size_t i = 0; i < num_layer_; ++i)
        attention_layers_[i] = new RelPositionAttentionLayer<T>(max_batch_size_,
                                                                max_seq_len_,
                                                                head_num_,
                                                                size_per_head_,
                                                                q_scaling_,
                                                                stream_,
                                                                cublas_wrapper_,
                                                                allocator_,
                                                                is_free_buffer_after_forward_,
                                                                sparse_);

    if (activation_type_ == ActivationType::Gelu) {
        ffn_layer_ = new GeluFfnLayer<T>(max_batch_size_,
                                         max_seq_len_,
                                         head_num_,
                                         size_per_head_,
                                         inter_size_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
    }
    else if (activation_type_ == ActivationType::Relu) {
        ffn_layer_ = new ReluFfnLayer<T>(max_batch_size_,
                                         max_seq_len_,
                                         head_num_,
                                         size_per_head_,
                                         inter_size_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
    }
    else if (activation_type_ == ActivationType::Silu) {
        if(int8_mode_==0)
        {
        ffn_layer_ = new SiluFfnLayer<T>(max_batch_size_,
                                         max_seq_len_,
                                         head_num_,
                                         size_per_head_,
                                         inter_size_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
        }
        else
        {
        ffn_layer_ = new SiluFfnLayer<T>(max_batch_size_,
                                         max_seq_len_,
                                         head_num_,
                                         size_per_head_,
                                         inter_size_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_,
                                         sparse_);
        }
    }

    conformer_conv_layer_ = new ConformerConvLayer<T>(max_batch_size_,
                                                      max_seq_len_,
                                                      head_num_,
                                                      size_per_head_,
                                                      stream_,
                                                      cublas_wrapper_,
                                                      allocator_,
                                                      is_free_buffer_after_forward_,
                                                      sparse_);
    allocateBuffer();
}

template<typename T>
WenetEncoder<T>::WenetEncoder(size_t max_batch_size,
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
                              ActivationType activation_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    d_model_(d_model),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    sm_(sm),
    q_scaling_(q_scaling),
    int8_mode_(int8_mode),
    sparse_(sparse),
    activation_type_(activation_type),
    vocab_size_(vocab_size),
    beam_width_(beam_width)
{
    initialize();
}

template<typename T>
WenetEncoder<T>::WenetEncoder(WenetEncoder<T> const& wenet_encoder):
    BaseLayer(wenet_encoder),
    max_batch_size_(wenet_encoder.max_batch_size_),
    max_seq_len_(wenet_encoder.max_seq_len_),
    head_num_(wenet_encoder.head_num_),
    size_per_head_(wenet_encoder.size_per_head_),
    inter_size_(wenet_encoder.inter_size_),
    d_model_(wenet_encoder.d_model_),
    hidden_units_(wenet_encoder.hidden_units_),
    num_layer_(wenet_encoder.num_layer_),
    sm_(wenet_encoder.sm_),
    q_scaling_(wenet_encoder.q_scaling_),
    int8_mode_(wenet_encoder.int8_mode_),
    sparse_(wenet_encoder.sparse_),
    activation_type_(wenet_encoder.activation_type_),
    vocab_size_(wenet_encoder.vocab_size_),
    beam_width_(wenet_encoder.beam_width_)
{
    initialize();
}

template<typename T>
WenetEncoder<T>::~WenetEncoder()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (auto& it : cuda_graph_pool_)
        delete it.second;
    cuda_graph_pool_.clear();

    for (size_t i = 0; i < num_layer_; ++i)
        delete attention_layers_[i];
    // delete attention_layer_;
    delete ffn_layer_;
    delete conformer_conv_layer_;

    check_cuda_error(cudaFreeHost(h_var_token_num_));
    check_cuda_error(cudaEventDestroy(stream2_finished_));
    check_cuda_error(cudaEventDestroy(stream_finished_));
    check_cuda_error(cudaStreamDestroy(stream2_));

    freeBuffer();
}

template<typename T>
void WenetEncoder<T>::setStream(cudaStream_t stream)
{
    for (size_t i = 0; i < num_layer_; ++i)
        attention_layers_[i]->setStream(stream);
    // attention_layer_->setStream(stream);
    ffn_layer_->setStream(stream);
    conformer_conv_layer_->setStream(stream);
    BaseLayer::setStream(stream);
}

template<typename T>
void WenetEncoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        size_t max_tensor_size = max_batch_size_ * max_seq_len_ * hidden_units_;

        attention_mask_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * 1 * max_seq_len_ * max_seq_len_, false);

        for (size_t li = 0; li < num_layer_; ++li) {
            pos_emb_repeated_[li] = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);
            pos_emb_cache_[li] = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);
        }

        token_num_ = (size_t*)allocator_->malloc(sizeof(size_t) * 1, false);
        padding_offset_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_ * max_seq_len_, false);

        bid_start_end_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_ * max_seq_len_ * 3, false);

        normed_from_tensor_ = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);

        ffn_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);
        normed_ffn_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);

        attn_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);
        normed_attn_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);

        conv_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);
        normed_conv_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);

        ffn2_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_tensor_size, false);

        ctc_lo_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * vocab_size_, false);

        log_softmax_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * vocab_size_, false);

        invokeTopkBeamSearch<float>(nullptr,
                                    topk_workspace_size_,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    max_batch_size_ * max_seq_len_,
                                    beam_width_,
                                    vocab_size_,
                                    0.0f,
                                    nullptr,
                                    stream_);
        topk_workspace_ = allocator_->malloc(topk_workspace_size_, false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
void WenetEncoder<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(batch_size <= max_batch_size_ && seq_len <= max_seq_len_);
    return;  // for cuda graph
    size_t cur_tensor_size = batch_size * seq_len * hidden_units_;
    size_t max_tensor_size = max_batch_size_ * max_seq_len_ * hidden_units_;

    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * 1 * seq_len * seq_len, false);

    for (size_t li = 0; li < num_layer_; ++li) {
        pos_emb_repeated_[li] = (T*)allocator_->reMalloc(pos_emb_repeated_[li], sizeof(T) * max_tensor_size, false);
        pos_emb_cache_[li] = (T*)allocator_->reMalloc(pos_emb_cache_[li], sizeof(T) * max_tensor_size, false);
    }

    token_num_ = (size_t*)allocator_->reMalloc(token_num_, sizeof(size_t) * 1, false);
    padding_offset_ = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false);
    bid_start_end_ = (int*)allocator_->reMalloc(bid_start_end_, sizeof(int) * batch_size * seq_len * 3, false);

    normed_from_tensor_ = (T*)allocator_->reMalloc(normed_from_tensor_, sizeof(T) * cur_tensor_size, false);

    ffn_out_buf_ = (T*)allocator_->reMalloc(ffn_out_buf_, sizeof(T) * cur_tensor_size, false);
    normed_ffn_out_buf_ = (T*)allocator_->reMalloc(normed_ffn_out_buf_, sizeof(T) * cur_tensor_size, false);

    attn_out_buf_ = (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * cur_tensor_size, false);
    normed_attn_out_buf_ = (T*)allocator_->reMalloc(normed_attn_out_buf_, sizeof(T) * cur_tensor_size, false);

    conv_out_buf_ = (T*)allocator_->reMalloc(conv_out_buf_, sizeof(T) * cur_tensor_size, false);
    normed_conv_out_buf_ = (T*)allocator_->reMalloc(normed_conv_out_buf_, sizeof(T) * cur_tensor_size, false);

    ffn2_out_buf_ = (T*)allocator_->reMalloc(ffn2_out_buf_, sizeof(T) * cur_tensor_size, false);

    ctc_lo_out_buf_ = (T*)allocator_->reMalloc(ctc_lo_out_buf_, sizeof(T) * batch_size * seq_len * vocab_size_, false);
    log_softmax_out_buf_ =
        (T*)allocator_->reMalloc(log_softmax_out_buf_, sizeof(T) * batch_size * seq_len * vocab_size_, false);

    invokeTopkBeamSearch<float>(nullptr,
                                topk_workspace_size_,
                                nullptr,
                                nullptr,
                                nullptr,
                                batch_size * seq_len,
                                beam_width_,
                                vocab_size_,
                                0.0f,
                                nullptr,
                                stream_);
    topk_workspace_ = allocator_->reMalloc(topk_workspace_, topk_workspace_size_, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void WenetEncoder<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free(attention_mask_);

        for (size_t li = 0; li < num_layer_; ++li) {
            allocator_->free(pos_emb_repeated_[li]);
            allocator_->free(pos_emb_cache_[li]);
        }

        allocator_->free(token_num_);
        allocator_->free(padding_offset_);
        allocator_->free(bid_start_end_);

        allocator_->free(normed_from_tensor_);

        allocator_->free(ffn_out_buf_);
        allocator_->free(normed_ffn_out_buf_);

        allocator_->free(attn_out_buf_);
        allocator_->free(normed_attn_out_buf_);

        allocator_->free(conv_out_buf_);
        allocator_->free(normed_conv_out_buf_);

        allocator_->free(ffn2_out_buf_);

        allocator_->free(ctc_lo_out_buf_);

        allocator_->free(log_softmax_out_buf_);

        allocator_->free(topk_workspace_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void WenetEncoder<T>::forward(std::vector<Tensor>* output_tensors,
                              const std::vector<Tensor>* input_tensors,
                              const WenetEncoderWeight<T>* encoder_weights)
{
    // input_tensors:
    //      input_ids [batch, seqlen]
    //      sequence_length [batch]
    // output tensors:
    //      output_hidden_state [batch, seqlen, hidden_units_]

    std::unordered_map<std::string, Tensor> input_tensors_map{
        {"input_hidden_state", input_tensors->at(0)},
        {"sequence_length", input_tensors->at(1)},
        {"pos_emb", input_tensors->at(2)},
        {"speech", input_tensors->at(3)},
    };

    std::unordered_map<std::string, Tensor> output_tensors_map{
        {"output_hidden_state", output_tensors->at(0)},
        {"encoder_out_lens", output_tensors->at(1)},
        {"ctc_log_probs", output_tensors->at(2)},
        {"beam_log_probs", output_tensors->at(3)},
        {"beam_log_probs_idx", output_tensors->at(4)},
    };
    forward(&output_tensors_map, &input_tensors_map, encoder_weights);
}

template<typename T>
void WenetEncoder<T>::forward(std::unordered_map<std::string, Tensor>* output_tensors,
                              const std::unordered_map<std::string, Tensor>* input_tensors,
                              const WenetEncoderWeight<T>* encoder_weights)
{
    // input_tensors:
    //      input_ids [batch, seqlen, hidden_size]
    //      sequence_length [batch]
    // output tensors:
    //      output_hidden_state [batch, seqlen, hidden_units_]

    // TODO(yuqingding):
    // done: invokeGeneralAddBiasResidualPreLayerNorm, invokeScaleAddBiasResidual
    // done: conv module + dp conv kernel
    // attn:
    // done: addmaskedsoftmax
    // done: addqkvpbiastranspose
    // done: allocate buf
    // done: build mask
    // done: repeat pos_emb
    // varlen + two stream parallel ???
    // logsoftmax + topk
    // merge ln2
    // cache pos_emb
    // tvm gemm for 1x3x256 * 256*256
    // float, half, bf16, int8, cudagraph, gemm config

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor input_hidden_state = input_tensors->at("input_hidden_state");
    const size_t batch_size = input_hidden_state.shape[0];
    const size_t seq_len = input_hidden_state.shape[1];
    FT_CHECK(input_tensors->size() == 4);
    Tensor seq_len_tensor = input_tensors->at("sequence_length");
    Tensor pos_emb_tensor = input_tensors->at("pos_emb");
    Tensor speech_tensor = input_tensors->at("speech");

    Tensor encoder_out_lens_tensor = output_tensors->at("encoder_out_lens");
    Tensor encoder_output_hidden_state_tensor = output_tensors->at("output_hidden_state");
    FT_CHECK(batch_size == seq_len_tensor.shape[0]);
    FT_CHECK(input_hidden_state.shape.size() == 3);
    FT_CHECK(seq_len_tensor.shape.size() == 1);
    allocateBuffer(batch_size, seq_len);

    const int* sequence_lengths_in = seq_len_tensor.getPtr<int>();
    int* sequence_lengths = encoder_out_lens_tensor.getPtr<int>();

    std::string cur_graph_key = FTCudaGraph::AppendShape2Key(speech_tensor.shape);
    FTCudaGraph* cur_graph_ptr = nullptr;

    // std::cout << cur_graph_key << "\t" << cur_graph_ptr << "\t" << stream_ << std::endl;
    /*
    std::cout << input_hidden_state.getPtr<void>() << ","
        << seq_len_tensor.getPtr<void>() << ","
        << pos_emb_tensor.getPtr<void>() << ","
        << speech_tensor.getPtr<void>() << "\t"
        << encoder_output_hidden_state_tensor.getPtr<void>() << ","
        << encoder_out_lens_tensor.getPtr<void>() << ","
        << std::endl;
*/
    if (is_enqueue_init_ && use_cuda_graph_) {
        FT_CHECK(is_free_buffer_after_forward_ == false);
        if (cuda_graph_pool_.find(cur_graph_key) == cuda_graph_pool_.end()) {
            cur_graph_ptr = new FTCudaGraph();
            // std::cout << cur_graph_key << "\t" << cur_graph_ptr << "\t" << stream_ << std::endl;
            cur_graph_ptr->beginCapture(stream_);
        }
        else {
            cur_graph_ptr = cuda_graph_pool_[cur_graph_key];
            // std::cout << cur_graph_key << "\t" << cur_graph_ptr << "\t" << stream_ << std::endl;
            cur_graph_ptr->launch(stream_);
            return;
        }
    }

    if (!is_enqueue_init_) {
        int n = hidden_units_;
        int m = max_seq_len_;
        int k = hidden_units_;

        for (size_t li = 0; li < num_layer_; ++li) {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  encoder_weights->encoder_layer_weights[li]->attention_weights.pos_weight.kernel,
                                  n,
                                  encoder_weights->emb_pe_weight,
                                  k,
                                  pos_emb_repeated_[li],
                                  n);
            // cublas_wrapper_->SpGemm(
            //     CUBLAS_OP_N, CUBLAS_OP_N, n, m_padded, k, attention_weights->pos_weight.sp_kernel, pos_emb, p_buf_);
            sync_check_cuda_error();

            invokeSNH2NSH(
                pos_emb_cache_[li], pos_emb_repeated_[li], 1, max_seq_len_, head_num_, size_per_head_, stream_);
            sync_check_cuda_error();

            const size_t pos_emb_size = m * k;
            for (size_t i = 0; i < max_batch_size_; i++) {
                check_cuda_error(cudaMemcpyAsync(pos_emb_repeated_[li] + i * pos_emb_size,
                                                 pos_emb_cache_[li],
                                                 pos_emb_size * sizeof(T),
                                                 ::cudaMemcpyDefault,
                                                 stream_));
                // cudaStreamSynchronize(stream_);
            }
            sync_check_cuda_error();
        }
    }

    invokeGetWenetOutLens(sequence_lengths, sequence_lengths_in, batch_size, speech_tensor.shape[1], stream_);
    sync_check_cuda_error();

    size_t h_token_num = batch_size * seq_len;
    T* input_ptr = input_hidden_state.getPtr<T>();
    T* output_ptr = encoder_output_hidden_state_tensor.getPtr<T>();
    float* ctc_log_probs_ptr = output_tensors->at("ctc_log_probs").getPtr<float>();
    // float* beam_log_probs_ptr = output_tensors->at("beam_log_probs").getPtr<float>();
    // int* beam_log_probs_idx_ptr = output_tensors->at("beam_log_probs_idx").getPtr<int>();

    invokeBuildEncoderAttentionMask(attention_mask_, sequence_lengths, batch_size, seq_len, stream_);
    sync_check_cuda_error();

    bool use_varlen = false;
    *h_var_token_num_ = h_token_num;
    if (use_varlen) {
        invokeGetPaddingOffset(token_num_, padding_offset_, sequence_lengths, batch_size, seq_len, stream_);
        sync_check_cuda_error();
        check_cuda_error(cudaEventRecord(stream_finished_, stream_));

        check_cuda_error(cudaStreamWaitEvent(stream2_, stream_finished_));
        check_cuda_error(
            cudaMemcpyAsync(h_var_token_num_, token_num_, sizeof(size_t), ::cudaMemcpyDeviceToHost, stream2_));
        sync_check_cuda_error();
        check_cuda_error(cudaEventRecord(stream2_finished_, stream2_));

        // check_cuda_error(cudaStreamWaitEvent(stream_, stream2_finished_));

        invokeGetBatchIDStartEnd(bid_start_end_, sequence_lengths, batch_size, seq_len, stream_);

        sync_check_cuda_error();
    }

    DataType data_type = getTensorType<T>();

    for (uint i = 0; i < num_layer_; i++) {
        const T* from_tensor = (const T*)(i == 0 ? input_ptr : output_ptr);
        T* out_tensor = output_ptr;

        invokeGeneralLayerNorm(normed_from_tensor_,
                               from_tensor,
                               encoder_weights->encoder_layer_weights[i]->norm_ff_macaron_weights.gamma,
                               encoder_weights->encoder_layer_weights[i]->norm_ff_macaron_weights.beta,
                               h_token_num,
                               hidden_units_,
                               stream_);

        sync_check_cuda_error();

        // feed_forward_macaron
        {
            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, normed_from_tensor_}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, ffn_out_buf_}};
            ffn_layer_->forward(&ffn_output_tensors,
                                &ffn_input_tensors,
                                &encoder_weights->encoder_layer_weights[i]->feed_forward_macaron_weights);
        }

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            ffn_out_buf_,
            normed_ffn_out_buf_,
            from_tensor,
            encoder_weights->encoder_layer_weights[i]->attn_layernorm_weights.gamma,
            encoder_weights->encoder_layer_weights[i]->attn_layernorm_weights.beta,
            encoder_weights->encoder_layer_weights[i]->feed_forward_macaron_weights.output_weight.bias,
            h_token_num,
            hidden_units_,
            stream_,
            2,
            0.5f,
            1.0f);
        sync_check_cuda_error();

        // attn
        {
            std::vector<Tensor> attn_input_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch_size * seq_len, hidden_units_},
                       normed_ffn_out_buf_},
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, 1, seq_len, seq_len}, attention_mask_},
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size * seq_len}, nullptr},
                Tensor{// MEMORY_GPU, data_type, std::vector<size_t>{batch_size * seq_len, hidden_units_},
                       // pos_emb_repeated_},
                       MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{max_batch_size_, head_num_, max_seq_len_, size_per_head_},
                       pos_emb_repeated_[i]},
            };

            std::vector<Tensor> attn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size * seq_len, hidden_units_}, attn_out_buf_}};

            attention_layers_[i]->forward(&attn_output_tensors,
                                          &attn_input_tensors,
                                          &encoder_weights->encoder_layer_weights[i]->attention_weights);
        }

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            attn_out_buf_,
            normed_attn_out_buf_,
            ffn_out_buf_,
            encoder_weights->encoder_layer_weights[i]->norm_conv_weights.gamma,
            encoder_weights->encoder_layer_weights[i]->norm_conv_weights.beta,
            encoder_weights->encoder_layer_weights[i]->attention_weights.attention_output_weight.bias,
            h_token_num,
            hidden_units_,
            stream_,
            2,
            1.0f,
            1.0f);
        sync_check_cuda_error();

        // conv
        {
            if (i == 0 && use_varlen) {
                check_cuda_error(cudaStreamWaitEvent(stream_, stream2_finished_));
            }
            std::vector<Tensor> conv_input_tensors{
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch_size, seq_len, hidden_units_},
                       normed_attn_out_buf_},
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, 1, seq_len, seq_len}, attention_mask_},
                Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{*h_var_token_num_}, padding_offset_},
                Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(*h_var_token_num_) * 3}, bid_start_end_}};

            std::vector<Tensor> conv_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batch_size, seq_len, hidden_units_}, conv_out_buf_}};
            conformer_conv_layer_->forward(&conv_output_tensors,
                                           &conv_input_tensors,
                                           &encoder_weights->encoder_layer_weights[i]->conv_module_weights);
        }

        T* bias_nullptr = nullptr;
        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            conv_out_buf_,
            normed_conv_out_buf_,
            attn_out_buf_,
            encoder_weights->encoder_layer_weights[i]->ffn_layernorm_weights.gamma,
            encoder_weights->encoder_layer_weights[i]->ffn_layernorm_weights.beta,
            bias_nullptr,  // encoder_weights->encoder_layer_weights[i]->conv_module_weights.pointwise_conv2_weight.bias,
            h_token_num,
            hidden_units_,
            stream_,
            2,
            1.0f,
            1.0f);
        sync_check_cuda_error();

        // ffn
        {
            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, normed_conv_out_buf_}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, ffn2_out_buf_}};
            ffn_layer_->forward(
                &ffn_output_tensors, &ffn_input_tensors, &encoder_weights->encoder_layer_weights[i]->ffn_weights);
        }

        invokeGeneralScaleAddBiasResidualPreLayerNorm(
            ffn2_out_buf_,
            out_tensor,
            conv_out_buf_,
            encoder_weights->encoder_layer_weights[i]->norm_final_weights.gamma,
            encoder_weights->encoder_layer_weights[i]->norm_final_weights.beta,
            encoder_weights->encoder_layer_weights[i]->ffn_weights.output_weight.bias,
            h_token_num,
            hidden_units_,
            stream_,
            2,
            0.5f,
            1.0f);
        sync_check_cuda_error();

    }
    invokeGeneralLayerNorm(output_ptr,
                           output_ptr,
                           encoder_weights->post_transformer_layernorm_weights.gamma,
                           encoder_weights->post_transformer_layernorm_weights.beta,
                           h_token_num,
                           hidden_units_,
                           stream_);
    sync_check_cuda_error();

    if (false) {  // NOTE(yuqingding): The test script does not need these outputs.
        int n = vocab_size_;
        int m = h_token_num;
        int k = hidden_units_;

        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              encoder_weights->ctc_lo_weight.kernel,
                              n,
                              output_ptr,
                              k,
                              ctc_lo_out_buf_,
                              n);

        // add bias and log_softmax
        invokeBiasLogSoftmax<T>(ctc_log_probs_ptr,
                                ctc_lo_out_buf_,
                                encoder_weights->ctc_lo_weight.bias,
                                sequence_lengths,
                                seq_len,
                                batch_size,
                                vocab_size_,
                                vocab_size_,
                                true,
                                stream_);
        sync_check_cuda_error();

        /*
        // TODO(yuqingding): This TopK function can't be used in this model, we need to rewrite the kernel.
        invokeTopkBeamSearch<float>(topk_workspace_,
                                    topk_workspace_size_,
                                    ctc_log_probs_ptr,
                                    beam_log_probs_idx_ptr,  // int* ids,
                                    nullptr,                 // const bool* finished,
                                    batch_size * seq_len,
                                    beam_width_,
                                    vocab_size_,
                                    0.0f,
                                    nullptr,  // const int* end_ids,
                                    stream_);
        sync_check_cuda_error();
        */
        // TODO(yuqingding): Gather val by topk id
        // ctc_log_probs_ptr + beam_log_probs_idx_ptr => beam_log_probs_ptr
    }
    if (is_enqueue_init_ && use_cuda_graph_) {
        if (cuda_graph_pool_.find(cur_graph_key) == cuda_graph_pool_.end()) {
            cur_graph_ptr->endCapture(stream_);
            cuda_graph_pool_[cur_graph_key] = cur_graph_ptr;
            // NOTE(yuqingding): If we don't rerun the stream, the result will be wrong.  Graph capture will destroy the
            // result???
            cur_graph_ptr->launch(stream_);
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
    if (!is_enqueue_init_) {
        is_enqueue_init_ = true;
    }
}

template class WenetEncoder<float>;
template class WenetEncoder<half>;

}  // namespace fastertransformer
