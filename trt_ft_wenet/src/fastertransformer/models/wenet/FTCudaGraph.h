/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "src/fastertransformer/utils/cuda_utils.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>

namespace fastertransformer {

class FTCudaGraph {
public:
    explicit FTCudaGraph() = default;

    FTCudaGraph(const FTCudaGraph&) = delete;

    FTCudaGraph& operator=(const FTCudaGraph&) = delete;

    FTCudaGraph(FTCudaGraph&&) = delete;

    FTCudaGraph& operator=(FTCudaGraph&&) = delete;

    ~FTCudaGraph()
    {
        if (mGraphExec) {
            check_cuda_error(cudaGraphExecDestroy(mGraphExec));
        }
    }

    void beginCapture(cudaStream_t stream)
    {
        check_cuda_error(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    }

    bool launch(cudaStream_t stream)
    {
        check_cuda_error(cudaGraphLaunch(mGraphExec, stream));
        return true;
    }

    void endCapture(cudaStream_t stream)
    {
        check_cuda_error(cudaStreamEndCapture(stream, &mGraph));
        check_cuda_error(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));
        check_cuda_error(cudaGraphDestroy(mGraph));
    }

    void endCaptureOnError(cudaStream_t stream)
    {
        const auto ret = cudaStreamEndCapture(stream, &mGraph);
        assert(ret == cudaErrorStreamCaptureInvalidated);
        assert(mGraph == nullptr);
        // Clean up the above CUDA error.
        cudaGetLastError();
        printf("The CUDA graph capture on the stream has failed.");
    }
    static std::string AppendShape2Key(std::vector<size_t> shape, std::string key = "")
    {
        std::ostringstream oss;
        oss << key;
        for (size_t i = 0; i < shape.size(); ++i)
            oss << "," << shape[i];
        return oss.str();
    }

private:
    cudaGraph_t mGraph{};
    cudaGraphExec_t mGraphExec{};
};

}  // namespace fastertransformer
