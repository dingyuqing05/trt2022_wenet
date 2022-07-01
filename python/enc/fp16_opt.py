#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

import sys
import onnx
import onnx_graphsurgeon as gs
import numpy as np

sys.path.append("/target/python")
import ching_utils as cu


def fix_Slice_79(graph):
    #not30 = get_node(graph, "Not", "Not_30")
    roots = cu.get_linear_graphs(graph, ["Not", "Slice"])
    for node in roots:
        not30 = node
        slice79 = cu.get_children(graph, not30, 0)[0]
        cu.insert_cast(graph, not30, slice79, 6)

    ###########################################

    #cu.cast_bool_before_not(graph)

    graph.cleanup().toposort()
    #exit()

def replace_encoder(graph):
    mul64 = cu.get_node(graph, "Mul", "Mul_64")
    #slice74 = cu.get_node(graph, "Slice", "Slice_74")
    add1979 = cu.get_linear_graphs(graph, ["Add", "encoder_out"])[0]
    cast1988 = cu.get_linear_graphs(graph, ["Cast", "encoder_out_lens"])[0]
    logsoftmax1987 = cu.get_linear_graphs(graph, ["LogSoftmax", "ctc_log_probs"])[0]
    topk1989 = cu.get_linear_graphs(graph, ["TopK", "beam_log_probs"])[0]

    in0 = mul64.outputs[0]
    in1 = cu.get_input(graph, "speech_lengths")
    #in1 = cu.get_output(graph, "encoder_out_lens")
    #in2 = slice74.outputs[0]
    in2 = in0  # use pos cache
    in3 = cu.get_input(graph, "speech")

    out0 = cu.get_output(graph, "encoder_out")
    out1 = cu.get_output(graph, "encoder_out_lens")
    out2 = cu.get_output(graph, "ctc_log_probs")
    out3 = cu.get_output(graph, "beam_log_probs")
    out4 = cu.get_output(graph, "beam_log_probs_idx")

    add1979.outputs = []
    cast1988.outputs = []
    logsoftmax1987.outputs = []
    topk1989.outputs = []

    new_node = gs.Node(op="WenetEncoderPlugin", inputs=[in0, in1, in2, in3], outputs=[out0, out1, out2, out3, out4])
    new_node.name = "WenetEncoder"
    new_node.attrs["max_batch_size"] = 16
    new_node.attrs["max_seq_len"] = 64
    new_node.attrs["head_num"] = 4
    new_node.attrs["size_per_head"] = 64
    new_node.attrs["inter_size"] = 2048
    new_node.attrs["d_model"] = 256
    new_node.attrs["num_layer"] = 12
    #new_node.attrs["sm"] = -1
    new_node.attrs["q_scaling"] = float(1.0)
    new_node.attrs["useFP16"] = True 
    new_node.attrs["int8_mode"] = 0
    new_node.attrs["vocab_size"] = 2982


   
    graph.nodes.append(new_node)

    graph.cleanup().toposort()


def findWeightInfInFP16(onnx_model, graph):
    ws = export_GetAllWeight(onnx_model, graph)
    for name in ws:
        x = np.amax(ws[name])
        if x > 65535:
            print(name, x)


if True:
    onnx_file_ori = "/target/python/encoder.onnx"
    onnx_file_opt = "/target/python/enc/encoder_opt.onnx"
    onnx_model = onnx.load(onnx_file_ori)
    
    graph = gs.import_onnx(onnx_model)
    #findWeightInfInFP16(onnx_model, graph)

    #fix_Slice_79(graph)

    replace_encoder(graph)

    #cu.merge_layer_norm(graph)

    #cu.merge_bias_skip_layernorm(graph)

    #cu.merge_mask_softmax(graph, "encoder_out_lens", "encoder_out_lens")

    cu.fold_const(graph)

    onnx.save(gs.export_onnx(graph), onnx_file_opt)

