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

#pragma once

#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/models/wenet/WenetEncoderLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct WenetEncoderWeight {

    WenetEncoderWeight() = default;
    WenetEncoderWeight(const size_t head_num,
                       const size_t size_per_head,
                       const size_t inter_size,
                       const size_t d_model,
                       const size_t vocab_size,
                       const size_t num_layer,
                       const size_t int8_mode);
    ~WenetEncoderWeight();
    WenetEncoderWeight(const WenetEncoderWeight& other);
    WenetEncoderWeight& operator=(const WenetEncoderWeight& other);

    std::vector<WenetEncoderLayerWeight<T>*> encoder_layer_weights;
    LayerNormWeight<T> post_transformer_layernorm_weights;
    DenseWeight<T> ctc_lo_weight;
    const T* emb_pe_weight = nullptr;

    void loadModel(std::string dir_path);

private:
    void setWeightPtr();
    void mallocWeights();
    void initialize();

    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t d_model_;
    size_t vocab_size_;
    size_t num_layer_;
    size_t int8_mode_;

    bool is_maintain_buffer = false;

    int real_weights_num_;

    const static int weights_num_ = 32;
    T* weights_ptr[weights_num_];
    size_t weights_size[weights_num_];
};

}  // namespace fastertransformer