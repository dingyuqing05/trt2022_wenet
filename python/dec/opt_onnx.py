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


def replace_decoder(graph):
    add167 = cu.get_node(graph, "Add", "Add_167")
    matmul1025 = cu.get_node(graph, "MatMul", "MatMul_1025")
    logsoftmax1027 = cu.get_node(graph, "LogSoftmax", "LogSoftmax_1027")


    in0 = add167.outputs[0]
    in1 = cu.get_input(graph, "hyps_lens_sos")
    in2 = cu.get_input(graph, "encoder_out")
    in3 = cu.get_input(graph, "encoder_out_lens")

    #out0 = gs.Variable(name="WenetDecoder_0_out", dtype=None)
    #matmul1025.inputs[0] = out0

    out0 = logsoftmax1027.outputs[0]
    logsoftmax1027.outputs = []

    new_node = gs.Node(op="WenetDecoderPlugin", inputs=[in0, in1, in2, in3], outputs=[out0,])
    new_node.name = "WenetDecoder"
    new_node.attrs["max_batch_size"] = 16 * 10
    new_node.attrs["max_seq_len"] = 256
    new_node.attrs["head_num"] = 4
    new_node.attrs["size_per_head"] = 64
    new_node.attrs["inter_size"] = 2048
    new_node.attrs["d_model"] = 256
    new_node.attrs["num_layer"] = 6
    new_node.attrs["q_scaling"] = 1.0
    #new_node.attrs["sm"] = -1
    new_node.attrs["useFP16"] = True

    graph.nodes.append(new_node)
    graph.cleanup().toposort()
    
def findWeightInfInFP16(onnx_model, graph):
    ws = export_GetAllWeight(onnx_model, graph)
    for name in ws:
        x = np.amax(ws[name])
        if x > 65535:
            print(name, x)


if True:
    onnx_file_ori = "/workspace/decoder.onnx"
    onnx_file_opt = "/target/python/dec/decoder_opt.onnx"
    onnx_model = onnx.load(onnx_file_ori)

    graph = gs.import_onnx(onnx_model)

    cu.set_dim_for_input(graph, "hyps_pad_sos_eos", 2, 64)
    cu.set_dim_for_output(graph, "decoder_out", 2, 64)

    cu.merge_layer_norm(graph)

    replace_decoder(graph)

    cu.fold_const(graph)
    
    onnx.save(gs.export_onnx(graph), onnx_file_opt)
