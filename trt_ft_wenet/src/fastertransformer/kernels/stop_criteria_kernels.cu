/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/stop_criteria_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

__global__ void stop_words_criterion(const int* output_ids,
                                     const int* parent_ids,
                                     const int* stop_words,
                                     bool* finished,
                                     size_t id_offset,
                                     size_t stop_words_len,
                                     int batch_size,
                                     int beam_width,
                                     int step)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y / beam_width;
    const int beam_idx = blockIdx.y % beam_width;

    const int* base_stop_words = stop_words + batch_idx * 2 * stop_words_len;
    const int* base_offsets = base_stop_words + stop_words_len;

    if (id >= stop_words_len || base_offsets[id] < 0) {
        return;
    }

    const int item_end = base_offsets[id];
    const int item_start = (id > 0) ? base_offsets[id - 1] : 0;
    const int item_size = item_end - item_start;

    /* The single-token case unconditionally bans the token */
    bool should_stop = false;

    /* Enough previously generated tokens to look for a match */
    if (step + 1 >= item_size) {
        should_stop = true;
        int parent_id = beam_idx;
        const bool gather_beam = beam_width > 1;

        for (int token_idx = item_size - 1; token_idx >= 0; token_idx--) {
            const int previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size * beam_width
                                                  + id_offset + batch_idx * beam_width + parent_id];

            if (previous_token != base_stop_words[item_start + token_idx]) {
                should_stop = false;
                break;
            }
            if (gather_beam) {
                parent_id = parent_ids[(step - (item_size - 1) + token_idx) * beam_width * batch_size + id_offset
                                       + batch_idx * beam_width + parent_id];

                if (parent_id < 0 || parent_id >= beam_width) {
                    should_stop = false;
                    break;
                }
            }
        }
    }

    if (should_stop) {
        finished[batch_idx * beam_width + beam_idx] = true;
    }
}

void invokeStopWordsCriterion(const int* output_ids,
                              const int* parent_ids,
                              const int* stop_words,
                              bool* finished,
                              size_t id_offset,
                              size_t stop_words_len,
                              int batch_size,
                              int beam_width,
                              int step,
                              cudaStream_t stream)
{
    dim3 block, grid;
    block.x = min(((stop_words_len + 32 - 1) / 32) * 32, 256UL);
    grid.x = (stop_words_len + block.x - 1) / block.x;
    grid.y = batch_size * beam_width;

    stop_words_criterion<<<grid, block, 0, stream>>>(
        output_ids, parent_ids, stop_words, finished, id_offset, stop_words_len, batch_size, beam_width, step);
    sync_check_cuda_error();
}

}  // namespace fastertransformer
