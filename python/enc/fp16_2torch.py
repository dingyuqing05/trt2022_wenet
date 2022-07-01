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
import os
import sys
import onnx
import onnx_graphsurgeon as gs
import numpy as np

sys.path.append("/target/python")
import ching_utils as cu


def export_GetAllWeight(model, gsg):
    exported_name = list()
    res = dict()
    for w in model.graph.initializer:
        if w.name.startswith("onnx::"):
            continue
        if len(str(w.name)) > 4:
            print("export ", w.name, w.dims, w.data_type)
            dtype = cu.onnx2np_type(w.data_type)
            res[w.name] = np.frombuffer(w.raw_data, dtype = dtype).reshape(w.dims)
            exported_name.append(w.name)
            if w.name.endswith("bias"):
                new_name = w.name[0:len(w.name)-4] + "weight"
                wname = cu.get_weight_by_bias(gsg, w.name)
                if wname == None:
                    continue
                w = cu.onnx_GetWeight(model, wname)
                dtype = cu.onnx2np_type(w.data_type)
                res[new_name] = np.frombuffer(w.raw_data, dtype = dtype).reshape(w.dims)
                res[new_name] = np.transpose(res[new_name], (1, 0))
                print("export ", w.name, w.dims, w.data_type, " -> ", new_name, res[new_name].shape)
                exported_name.append(w.name)
    def get_not(model, exported_name):    
        not_name = list()
        for w in model.graph.initializer:
            if w.name not in exported_name:
                #print("not export ", w.name, w.dims, w.data_type)
                not_name.append(w.name)
        return not_name
    
    not_name = get_not(model, exported_name)
    cur_idx = 0
    for w in model.graph.initializer:
        if w.name in not_name and len(w.dims) ==2 and w.dims[0] == 256 and w.dims[1] == 256:
            for node in gsg.nodes:
                if node.op == "MatMul" and node.inputs[1].name == w.name:
                    new_name = "encoder.encoders." + str(cur_idx) + ".self_attn.linear_pos.weight"
                    #print("export ", w.name, w.dims, w.data_type, " -> ", new_name)
                    dtype = cu.onnx2np_type(w.data_type)
                    res[new_name] = np.frombuffer(w.raw_data, dtype = dtype).reshape(w.dims)
                    res[new_name] = np.transpose(res[new_name], (1, 0))
                    print("export ", w.name, w.dims, w.data_type, " -> ", new_name, res[new_name].shape)
                    exported_name.append(w.name)
                    cur_idx += 1

    not_name = get_not(model, exported_name)
    cur_idx = 0
    for w in model.graph.initializer:
        if w.name in not_name and len(w.dims) == 3:
            for node in gsg.nodes:
                if node.op == "Conv" and len(node.inputs) == 3 and node.inputs[1].name == w.name:
                    new_name = "encoder.encoders." + str(cur_idx) + ".conv_module.depthwise_conv.weight"
                    print("export ", w.name, w.dims, w.data_type, " -> ", new_name)
                    dtype = cu.onnx2np_type(w.data_type)
                    res[new_name] = np.frombuffer(w.raw_data, dtype = dtype).reshape(w.dims)
                    exported_name.append(w.name)

                    bname = node.inputs[2].name
                    w = cu.onnx_GetWeight(model, bname)
                    new_name = "encoder.encoders." + str(cur_idx) + ".conv_module.depthwise_conv.bias"
                    print("export ", w.name, w.dims, w.data_type, " -> ", new_name)
                    dtype = cu.onnx2np_type(w.data_type)
                    res[new_name] = np.frombuffer(w.raw_data, dtype = dtype).reshape(w.dims)
                    exported_name.append(w.name)

                    cur_idx += 1

    if True:
        for node in gsg.nodes:
            if node.op == "Slice":
                slice74 = node
                pnode = cu.get_parent(graph, slice74, 0)
                if pnode.op != "Constant":
                    continue
                w = cu.GetConstValue(graph, pnode)
                assert len(w.shape) == 3
                #w = np.asarray(w)
                old_name = pnode.outputs[0].name
                new_name = "encoder.embed.pe"
                print("export ", old_name, w.shape, w.dtype, " -> ", new_name)
                res[new_name] = w
                #print(type(w))
                print(w)
                exported_name.append(old_name)
                break

    if True:
        not_name = get_not(model, exported_name)
        for w in model.graph.initializer:
            if w.name in not_name:
                print("not export ", w.name, w.dims, w.data_type)

    amax_file = "/target/python/wenet_amax.npy"
    if os.path.exists(amax_file):
        am = np.load(amax_file, allow_pickle='TRUE')
        am = am.item()
        res.update(am)

    np.save("/target/python/enc/encoder.npy", res)
    #print(res["encoder.encoders.0.norm_ff_macaron.weight"])
    #print(res["encoder.encoders.0.self_attn.linear_pos.weight"])
    return res


if True:
    onnx_file_ori = "/target/python/encoder.onnx"
    #onnx_file_opt = "/target/python/enc/encoder_opt.onnx"
    onnx_model = onnx.load(onnx_file_ori)
    #all_ws = onnx_GetAllWeight(onnx_model)
    
    graph = gs.import_onnx(onnx_model)
    export_GetAllWeight(onnx_model, graph)


