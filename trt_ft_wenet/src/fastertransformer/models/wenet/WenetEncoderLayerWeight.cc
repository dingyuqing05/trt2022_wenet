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

#include "src/fastertransformer/models/wenet/WenetEncoderLayerWeight.h"
#include "src/fastertransformer/models/wenet/WenetKernels.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/kernels/quantize_weight.h"


namespace fastertransformer {

namespace {

template<typename T_IN, typename T_OUT>
inline T_OUT convert_to_type(T_IN val)
{
    return (T_OUT)val;
}

#ifdef ENABLE_BF16
template<>
inline __nv_bfloat16 convert_to_type<float, __nv_bfloat16>(float val)
{
    return __float2bfloat16(val);
}

template<>
inline __nv_bfloat16 convert_to_type<half, __nv_bfloat16>(half val)
{
    return __float2bfloat16(__half2float(val));
}

template<>
inline float convert_to_type<__nv_bfloat16, float>(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template<>
inline half convert_to_type<__nv_bfloat16, half>(__nv_bfloat16 val)
{
    return __float2half(__bfloat162float(val));
}
#endif  // ENABLE_BF16


template<typename T, typename T_IN>
int loadAMaxFromBinFunc(T* ptr,std::string filename)
{
    size_t size = 1;
    std::vector<T_IN> host_array(size);
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        FT_LOG_WARNING("file %s cannot be opened, loading model fails! \n", filename.c_str());
        return 0;
    }

    size_t loaded_data_size = sizeof(T_IN) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    FT_LOG_DEBUG("Read " + std::to_string(loaded_data_size) + " bytes from " + filename);
    in.read((char*)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        FT_LOG_WARNING("file %s only has %ld, but request %ld, loading model fails! \n",
                       filename.c_str(),
                       in_get_size,
                       loaded_data_size);
        return 0;
    }
    *ptr = convert_to_type<T_IN, T>(host_array[0]);
    in.close();
    return 0;
}

}


template<typename T>
WenetEncoderLayerWeight<T>::WenetEncoderLayerWeight(const size_t layer_id,
                                                    const size_t head_num,
                                                    const size_t size_per_head,
                                                    const size_t inter_size,
                                                    const size_t int8_mode):
    layer_id_(layer_id),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    int8_mode_(int8_mode),
    real_weights_num_(35)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    setWeightPtr();
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetEncoderLayerWeight<T>::initialize()
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    int hidden_size = head_num_ * size_per_head_;
    int idx = 0;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = inter_size_;
    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    /*
    encoder.encoders.0.conv_module.pointwise_conv1.weight torch.Size([512, 256, 1])
    encoder.encoders.0.conv_module.pointwise_conv1.bias torch.Size([512])
    encoder.encoders.0.conv_module.depthwise_conv.weight torch.Size([256, 1, 15])
    encoder.encoders.0.conv_module.depthwise_conv.bias torch.Size([256])
    encoder.encoders.0.conv_module.pointwise_conv2.weight torch.Size([256, 256, 1])
    encoder.encoders.0.conv_module.pointwise_conv2.bias torch.Size([256])
    */
    weights_size[idx++] = hidden_size * 2 * hidden_size;
    weights_size[idx++] = hidden_size * 2;
    weights_size[idx++] = hidden_size * 1 * 15;
    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size * hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = inter_size_;
    weights_size[idx++] = hidden_size * inter_size_;
    weights_size[idx++] = hidden_size;

    weights_size[idx++] = hidden_size;
    weights_size[idx++] = hidden_size;

    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetEncoderLayerWeight<T>::~WenetEncoderLayerWeight()
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    if (is_maintain_buffer == true) {

        norm_ff_macaron_weights.gamma = nullptr;
        norm_ff_macaron_weights.beta = nullptr;

        feed_forward_macaron_weights.intermediate_weight.kernel = nullptr;
        feed_forward_macaron_weights.intermediate_weight.bias = nullptr;
        feed_forward_macaron_weights.output_weight.kernel = nullptr;
        feed_forward_macaron_weights.output_weight.bias = nullptr;

        attn_layernorm_weights.gamma = nullptr;
        attn_layernorm_weights.beta = nullptr;

        attention_weights.query_weight.kernel = nullptr;
        attention_weights.query_weight.bias = nullptr;
        attention_weights.key_weight.kernel = nullptr;
        attention_weights.key_weight.bias = nullptr;
        attention_weights.value_weight.kernel = nullptr;
        attention_weights.value_weight.bias = nullptr;
        attention_weights.attention_output_weight.kernel = nullptr;
        attention_weights.attention_output_weight.bias = nullptr;
        attention_weights.pos_weight.kernel = nullptr;
        attention_weights.pos_weight.bias = nullptr;
        attention_weights.pos_bias_u = nullptr;
        attention_weights.pos_bias_v = nullptr;

        // todo: ln + conv
        norm_conv_weights.gamma = nullptr;
        norm_conv_weights.beta = nullptr;

        conv_module_weights.pointwise_conv1_weight.kernel = nullptr;
        conv_module_weights.pointwise_conv1_weight.bias = nullptr;
        conv_module_weights.depthwise_conv_weight.kernel = nullptr;
        conv_module_weights.depthwise_conv_weight.bias = nullptr;
        conv_module_weights.pointwise_conv2_weight.kernel = nullptr;
        conv_module_weights.pointwise_conv2_weight.bias = nullptr;

        ffn_layernorm_weights.gamma = nullptr;
        ffn_layernorm_weights.beta = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias = nullptr;
        ffn_weights.output_weight.kernel = nullptr;
        ffn_weights.output_weight.bias = nullptr;

        norm_final_weights.gamma = nullptr;
        norm_final_weights.beta = nullptr;

        /*
            if (is_maintain_sp_buffer == true) {
                for (int i = 0; i < 6; i++) {
                    deviceFree(sp_weights_ptr[i]);
                }
                attention_weights.query_weight.sp_kernel = nullptr;
                attention_weights.key_weight.sp_kernel = nullptr;
                attention_weights.value_weight.sp_kernel = nullptr;
                attention_weights.attention_output_weight.sp_kernel = nullptr;
                ffn_weights.intermediate_weight.sp_kernel = nullptr;
                ffn_weights.output_weight.sp_kernel = nullptr;
                is_maintain_sp_buffer = false;
            }
        */
        FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
    }
}
template<typename T>
WenetEncoderLayerWeight<T>::WenetEncoderLayerWeight(const WenetEncoderLayerWeight& other):
    layer_id_(other.layer_id_),
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    inter_size_(other.inter_size_),
    int8_mode_(other.int8_mode_),
    real_weights_num_(35)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
WenetEncoderLayerWeight<T>& WenetEncoderLayerWeight<T>::operator=(const WenetEncoderLayerWeight<T>& other)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    layer_id_ = other.layer_id_;
    head_num_ = other.head_num_;
    size_per_head_ = other.size_per_head_;
    inter_size_ = other.inter_size_;
    int8_mode_ = other.int8_mode_;
    real_weights_num_ = other.real_weights_num_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");

    return *this;
}
/*
#ifdef SPARSITY_ENABLED
template<typename T>
void WenetEncoderLayerWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper, int hidden_dim)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    int inter_size = hidden_dim * 4;
    deviceMalloc(&sp_weights_ptr[0], weights_size[0]);
    deviceMalloc(&sp_weights_ptr[1], weights_size[1]);
    deviceMalloc(&sp_weights_ptr[2], weights_size[2]);
    deviceMalloc(&sp_weights_ptr[3], weights_size[3]);
    deviceMalloc(&sp_weights_ptr[4], weights_size[5]);
    deviceMalloc(&sp_weights_ptr[5], weights_size[6]);

    cublas_wrapper.compressMatrix(attention_weights.query_weight.kernel,
                                  sp_weights_ptr[0],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights.key_weight.kernel,
                                  sp_weights_ptr[1],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights.value_weight.kernel,
                                  sp_weights_ptr[2],
                                  d_model_,
                                  (head_num_ / tensor_para_size_) * size_per_head_);
    cublas_wrapper.compressMatrix(attention_weights.attention_output_weight.kernel,
                                  sp_weights_ptr[3],
                                  (head_num_ / tensor_para_size_) * size_per_head_,
                                  d_model_);
    cublas_wrapper.compressMatrix(
        ffn_weights.intermediate_weight.kernel, sp_weights_ptr[4], inter_size / tensor_para_size_, d_model_);
    cublas_wrapper.compressMatrix(
        ffn_weights.output_weight.kernel, sp_weights_ptr[5], d_model_, inter_size / tensor_para_size_);
    attention_weights.query_weight.sp_kernel = sp_weights_ptr[0];
    attention_weights.key_weight.sp_kernel = sp_weights_ptr[1];
    attention_weights.value_weight.sp_kernel = sp_weights_ptr[2];
    attention_weights.attention_output_weight.sp_kernel = sp_weights_ptr[3];
    ffn_weights.intermediate_weight.sp_kernel = sp_weights_ptr[4];
    ffn_weights.output_weight.sp_kernel = sp_weights_ptr[5];
    is_maintain_sp_buffer = true;
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}
#endif
*/
template<typename T>
void WenetEncoderLayerWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    int qidx = 0;
    int idx = 0;
    norm_ff_macaron_weights.gamma = weights_ptr[idx++];
    norm_ff_macaron_weights.beta = weights_ptr[idx++];

    feed_forward_macaron_weights.intermediate_weight.kernel = weights_ptr[idx++];
    feed_forward_macaron_weights.intermediate_weight.bias = weights_ptr[idx++];
    feed_forward_macaron_weights.intermediate_weight.kernel_qscale = qscales_[qidx++];
    feed_forward_macaron_weights.intermediate_weight.in_qscale = qscales_[qidx++];
    feed_forward_macaron_weights.intermediate_weight.out_qscale = qscales_[qidx++];

    feed_forward_macaron_weights.output_weight.kernel = weights_ptr[idx++];
    feed_forward_macaron_weights.output_weight.bias = weights_ptr[idx++];
    feed_forward_macaron_weights.output_weight.kernel_qscale = qscales_[qidx++];
    feed_forward_macaron_weights.output_weight.in_qscale = qscales_[qidx++];
    feed_forward_macaron_weights.output_weight.out_qscale = qscales_[qidx++];

    attn_layernorm_weights.gamma = weights_ptr[idx++];
    attn_layernorm_weights.beta = weights_ptr[idx++];

    attention_weights.query_weight.kernel = weights_ptr[idx++];
    attention_weights.query_weight.bias = weights_ptr[idx++];
    attention_weights.key_weight.kernel = weights_ptr[idx++];
    attention_weights.key_weight.bias = weights_ptr[idx++];
    attention_weights.value_weight.kernel = weights_ptr[idx++];
    attention_weights.value_weight.bias = weights_ptr[idx++];
    attention_weights.attention_output_weight.kernel = weights_ptr[idx++];
    attention_weights.attention_output_weight.bias = weights_ptr[idx++];
    attention_weights.pos_weight.kernel = weights_ptr[idx++];
    // attention_weights.pos_weight.bias = weights_ptr[idx++];
    attention_weights.pos_bias_u = weights_ptr[idx++];
    attention_weights.pos_bias_v = weights_ptr[idx++];

    // todo: ln + conv
    norm_conv_weights.gamma = weights_ptr[idx++];
    norm_conv_weights.beta = weights_ptr[idx++];

    conv_module_weights.pointwise_conv1_weight.kernel = weights_ptr[idx++];
    conv_module_weights.pointwise_conv1_weight.bias = weights_ptr[idx++];
    conv_module_weights.depthwise_conv_weight.kernel = weights_ptr[idx++];
    conv_module_weights.depthwise_conv_weight.bias = weights_ptr[idx++];
    conv_module_weights.pointwise_conv2_weight.kernel = weights_ptr[idx++];
    conv_module_weights.pointwise_conv2_weight.bias = weights_ptr[idx++];

    ffn_layernorm_weights.gamma = weights_ptr[idx++];
    ffn_layernorm_weights.beta = weights_ptr[idx++];

    ffn_weights.intermediate_weight.kernel = weights_ptr[idx++];
    ffn_weights.intermediate_weight.bias = weights_ptr[idx++];
    ffn_weights.intermediate_weight.kernel_qscale = qscales_[qidx++];
    ffn_weights.intermediate_weight.in_qscale = qscales_[qidx++];
    ffn_weights.intermediate_weight.out_qscale = qscales_[qidx++];

    ffn_weights.output_weight.kernel = weights_ptr[idx++];
    ffn_weights.output_weight.bias = weights_ptr[idx++];
    ffn_weights.output_weight.kernel_qscale = qscales_[qidx++];
    ffn_weights.output_weight.in_qscale = qscales_[qidx++];
    ffn_weights.output_weight.out_qscale = qscales_[qidx++];

    norm_final_weights.gamma = weights_ptr[idx++];
    norm_final_weights.beta = weights_ptr[idx++];

    is_maintain_buffer = true;
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetEncoderLayerWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void WenetEncoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " start");

    FT_CHECK(is_maintain_buffer == true);

    std::vector<std::string> weights_name;
    std::string name_prefix = "encoder.encoders.";
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_ff_macaron.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_ff_macaron.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_1.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_1.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_2.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_2.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_mha.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_mha.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_q.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_q.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_k.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_k.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_v.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_v.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_out.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_out.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.linear_pos.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.pos_bias_u");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".self_attn.pos_bias_v");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_conv.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_conv.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.pointwise_conv1.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.pointwise_conv1.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.depthwise_conv.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.depthwise_conv.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.pointwise_conv2.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".conv_module.pointwise_conv2.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_ff.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_ff.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_1.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_1.bias");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_2.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_2.bias");

    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_final.weight");
    weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".norm_final.bias");

    std::vector<std::string> int8_weights_name;
    int8_weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_1.weight");
    int8_weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward_macaron.w_2.weight");
    int8_weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_1.weight");
    int8_weights_name.push_back(name_prefix + std::to_string(layer_id_) + ".feed_forward.w_2.weight");


    bool use_int8 = int8_mode_ != 0;
    T* int8_buffer_ = nullptr;
    int qi = 0;
    for (size_t i = 0; i < weights_name.size(); ++i) {
        T* cur_ptr = weights_ptr[i];
        bool is_cur_int8_weight = false;
        for(auto& tgt : int8_weights_name)
            if(tgt==weights_name[i])
                is_cur_int8_weight = true;
        if(use_int8 && is_cur_int8_weight)
        {
            deviceMalloc(&int8_buffer_, weights_size[i]);
            cur_ptr = int8_buffer_;
        }
        loadWeightFromBin<T>(
            cur_ptr, {(int)weights_size[i]}, dir_path + weights_name[i] + ".bin", model_file_type);

        if(use_int8 && is_cur_int8_weight)
        {
            // load amax
            std::string cur_prefix = weights_name[i].substr(0,weights_name[i].length()-6);

            std::string aname = weights_name[i] + "._amax";
            loadAMaxFromBinFunc<float, float>(&(qscales_[qi++]),  aname);

            aname = cur_prefix + "input_quantizer._amax";
            loadAMaxFromBinFunc<float, float>(&(qscales_[qi++]),  aname);

            aname = cur_prefix + "out_quantizer._amax";
            loadAMaxFromBinFunc<float, float>(&(qscales_[qi++]),  aname);

            float wq = qscales_[qi-3];
            int n = head_num_ * size_per_head_;
            int k = inter_size_;
            if((qi/3)%2==1)
                std::swap(n, k);
            
            invokeQuantizeWeight((int8_t*)weights_ptr[i],
                                    int8_buffer_,
                                    &wq,
                                    n,
                                    k,
                                    2,
                                    0,
                                    false);
            deviceFree(int8_buffer_);
            cudaDeviceSynchronize();
        }
    }
    for(int i = 0;i<qi;++i)
    {
        qscales_[i] = 127.0/qscales_[i];
    }
    FT_LOG_DEBUG("WenetEncoderLayerWeight " + std::string(__func__) + " end");
}

template struct WenetEncoderLayerWeight<float>;
template struct WenetEncoderLayerWeight<half>;

}  // namespace fastertransformer
