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

import onnx
import onnx_graphsurgeon as gs
import numpy as np

def get_input(graph, name):
    for node in graph.inputs:
        if node.name == name:
            return node
    return None

def get_output(graph, name):
    for node in graph.outputs:
        if node.name == name:
            return node
    return None

def get_node(graph, op, name):
    for node in graph.nodes:
        if node.op == op and node.name == name:
            return node
    return None

def get_linear_graphs(graph, ops):
    res = list()
    for node in graph.nodes:
        if node.op == ops[0] or node.name == ops[0]:
            cur = node
            find = True
            for i in range(len(ops)):
                if i == 0:
                    continue
                children = get_children(graph, cur, 0)
                find_child = False
                for c in children:
                    if c.op == ops[i] or c.name == ops[i]:
                        find_child = True
                        cur = c
                        break
                if not find_child:
                    for c in graph.outputs:
                        if c.name == ops[i] and c.name == cur.outputs[0].name:
                            find_child = True
                if not find_child:
                    find = False
                    break
            if find:
                res.append(node)
    return res 

def get_parent(graph, cur_node, parrent_pos):
    in0 = cur_node.inputs[parrent_pos]
    for node in graph.nodes:
        for out in node.outputs:
            if out == in0:
                return node
    return None

def get_producer(graph, tensor):
    in0 = tensor
    for node in graph.nodes:
        for out in node.outputs:
            if out == in0:
                return node
    return None

def count_consumer(graph, cur_node, child_pos):
    tgt = cur_node.outputs[child_pos]
    res = 0
    for node in graph.nodes:
        for ins in node.inputs:
            if ins == tgt:
                res += 1
    for ins in graph.outputs:
        if ins.name == tgt.name:
            res += 1
    return res

def get_children(graph, cur_node, child_pos):
    tgt = cur_node.outputs[child_pos]
    res = list()
    for node in graph.nodes:
        for ins in node.inputs:
            if ins == tgt:
                res.append(node)
    return res

def get_weight_by_bias(graph, bname):
    for node in graph.nodes:
        if node.op == "Add" and node.inputs[0].name == bname:
            pnode = get_parent(graph, node, 1)
            if pnode is not None and pnode.op == "MatMul":
                return pnode.inputs[1].name
    return None

def insert_cast(graph, node1, node2, to):
    op_name = "cast_" + node1.name + "_" + node2.name
    out_name = op_name + "_out"
    cast_out = gs.Variable(name=out_name, dtype=None)
    cast_node = gs.Node(op="Cast", inputs=[node1.outputs[0],], outputs=[cast_out,])
    cast_node.name = op_name
    cast_node.attrs["to"] = to

    node2.inputs[0] = cast_out
    graph.nodes.append(cast_node)

def cast_bool_before_not(graph):
    for node in graph.nodes:
        if node.op == "Not":
            node_prev = get_parent(graph, node, 0)
            insert_cast(graph, node_prev, node, 9)

def onnx_GetAllWeight(model):
    for w in model.graph.initializer:
        print(w.name, w.dims)
    return model.graph.initializer

def onnx2np_type(dtype):
    maps = {
            1: np.float32,
            6: np.int32,
            7: np.int64
    }
    return maps[dtype]

def onnx_GetWeight(model, name):
    for w in model.graph.initializer:
        if w.name == name:
            return w
    return None

def onnx_GetInit(model, name):
    for w in model.graph.initializer:
        #print(w.name, name)
        if w.name == name:
            dtype = cu.onnx2np_type(w.data_type)
            return np.frombuffer(w.raw_data, dtype = dtype).reshape(w.dims)
    return None

def GetConstValue(graph, node):
    return node.attrs['value'].values

def set_dim_for_input(graph, name, dim, val):
    for cur_in in graph.inputs:
        if cur_in.name == name:
            cur_in.shape[dim] = val;
    #print(graph.inputs)

def set_dim_for_output(graph, name, dim, val):
    for cur in graph.outputs:
        if cur.name == name:
            cur.shape[dim] = val;
    #print(graph.outputs)

def set_node_to_output(graph, op, name):
    node = get_node(graph, op, name)
    node.outputs[0].dtype = np.float32
    graph.outputs = [graph.outputs[0], node.outputs[0]]

def fold_const(graph):
    old_cc = len(graph.nodes)
    graph.fold_constants()
    graph.cleanup().toposort()
    new_cc = len(graph.nodes)
    print("fold const:", old_cc, " -> ", new_cc, " = ", old_cc - new_cc)

def remove_node_if_1consumer(rm_list, graph, tmp, cond):
    if (len(tmp.outputs) == 1 and count_consumer(graph, tmp, 0) == 1 and cond):
        rm_list.append(tmp)
        return True
    else:
        print("WARNING:", tmp.name, " will be reserved")
        return False

def merge_layer_norm(graph):
    ln_cc = 0
    for add in graph.nodes:
        if add.op == "Add":
            addp = get_parent(graph, add, 0)
            if addp is None or addp.op != "Mul":
                continue
            gamma = addp.inputs[1]
            addp = get_parent(graph, addp, 0)
            if addp is None or addp.op != "Div":
                continue
            addp = get_parent(graph, addp, 1)
            if addp is None or addp.op != "Sqrt":
                continue
            addp = get_parent(graph, addp, 0)
            if addp is None or addp.op != "Add":
                continue
            addp = get_parent(graph, addp, 0)
            if addp is None or addp.op != "ReduceMean":
                continue
            addp = get_parent(graph, addp, 0)
            if addp is None or addp.op != "Pow":
                continue
            addp = get_parent(graph, addp, 0)
            if addp is None or addp.op != "Sub":
                continue
            addp = get_parent(graph, addp, 1)
            if addp is None or addp.op != "ReduceMean":
                continue

            beta = add.inputs[1]
            cur_out = add.outputs[0]
            cur_in = addp.inputs[0]
            addp.inputs = []
            add.outputs = []

            ln_node = gs.Node(op="CustomLayerNormPluginDynamic", inputs=[cur_in, gamma, beta], outputs=[cur_out])
            graph.nodes.append(ln_node)
            ln_cc += 1
    print("Merge layer_norm count: ", ln_cc)
    graph.cleanup().toposort()


def get_input_value(graph, model, node, in_pos):
    tmp = onnx_GetInit(model, node.inputs[in_pos].name)
    if tmp is None:
        tmp = get_parent(graph, node, 1)
        tmp = GetConstValue(graph, tmp)
    return tmp





