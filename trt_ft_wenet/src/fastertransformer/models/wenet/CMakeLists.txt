# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(WenetEncoder STATIC WenetEncoder.cc 
								WenetEncoderWeight.cc 
								WenetEncoderLayerWeight.cc 
								WenetKernels.cu
								SiluFfnLayer.cc
								RelPositionAttentionLayer.cc
								ConformerConvLayer.cc
								SelfAttentionKernels.cu
								)
set_property(TARGET WenetEncoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET WenetEncoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(WenetEncoder PUBLIC -lcudart 
	bert_preprocess_kernels 
	cublasMMWrapper
	FfnLayer
	UnfusedAttentionLayer
	layernorm_kernels
	add_residual_kernels
	beam_search_topk_kernels
	quantize_weight
	cublasINT8MMWrapper
	)



add_library(WenetDecoder STATIC WenetDecoder.cc 
								WenetDecoderWeight.cc
								WenetDecoderLayerWeight.cc
								WenetKernels.cu
								MultiHeadedAttentionLayer.cc
								CrossAttentionKernels.cu
								)
set_property(TARGET WenetDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET WenetDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(WenetDecoder PUBLIC -lcudart 
	bert_preprocess_kernels 
	cublasMMWrapper
	FfnLayer
	UnfusedAttentionLayer
	layernorm_kernels
	add_residual_kernels
	quantize_weight
	cublasINT8MMWrapper
	)

#add_executable(t5_gemm t5_gemm.cc)
#target_link_libraries(t5_gemm PUBLIC -lcudart t5_gemm_func memory_utils)
