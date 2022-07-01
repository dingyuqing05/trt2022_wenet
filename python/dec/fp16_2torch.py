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
    if True:
        not_name = get_not(model, exported_name)
        for w in model.graph.initializer:
            if w.name in not_name:
                dtype = cu.onnx2np_type(w.data_type)
                cur = np.frombuffer(w.raw_data, dtype = dtype).reshape(w.dims)
                print("not export ", w.name, w.dims, w.data_type, cur)
    
    np.save("/target/python/dec/decoder.npy", res)
    return res
    
if True:
    onnx_file_ori = "/target/python/decoder.onnx"
    onnx_file_opt = "/target/python/dec/decoder_opt.onnx"
    onnx_model = onnx.load(onnx_file_ori)
    #all_ws = onnx_GetAllWeight(onnx_model)

    graph = gs.import_onnx(onnx_model)
    export_GetAllWeight(onnx_model, graph)

