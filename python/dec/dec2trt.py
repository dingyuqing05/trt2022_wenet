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
import numpy as np
import ctypes
from glob import glob 
#from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt


#onnxFile = "/workspace/decoder.onnx"
onnxFile = "/target/python/dec/decoder_opt.onnx"
trtFile = "/target/decoder.plan"
timeCacheFile = "/target/python/dec/tc_fp16_ln16.cache"
useTimeCache = False

np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

planFilePath = "/target/"
soFileList = glob(planFilePath + "*.so")
if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!!!!!!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

#logger = trt.Logger(trt.Logger.ERROR)
#logger = trt.Logger(trt.Logger.VERBOSE)
logger = trt.Logger(trt.Logger.INFO)

def make_logsoftmax_fp32(network):
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if str(layer.type)[10:] == "SOFTMAX":  # LogSoftmax_1027
            layer.precision = trt.DataType.FLOAT
            if i < network.num_layers:
                next_layer = network.get_layer(i+1)
                next_layer.precision = trt.DataType.FLOAT
    #exit(0)

def print_net(network):
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(i,"%s,in=%d,out=%d,%s"%(str(layer.type)[10:],layer.num_inputs,layer.num_outputs,layer.name))
        for j in range(layer.num_inputs):
            tensor  =layer.get_input(j)
            if tensor == None:
                print("\tInput  %2d:"%j,"None")
            else:
                print("\tInput  %2d:%s,%s,%s"%(j,tensor.shape,str(tensor.dtype)[9:],tensor.name))
        for j in range(layer.num_outputs):
            tensor  =layer.get_output(j)
            if tensor == None:
                print("\tOutput %2d:"%j,"None")
            else:
                print("\tOutput %2d:%s,%s,%s"%(j,tensor.shape,str(tensor.dtype)[9:],tensor.name))
    exit(0)

if True:
    timeCache = b""
    if useTimeCache and os.path.isfile(timeCacheFile):
        with open(timeCacheFile, 'rb') as f:
            timeCache = f.read()
        if timeCache == None:
            print("Failed getting serialized timing cache!")
        print("Succeeded getting serialized timing cache!")

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.flags = 1 << int(trt.BuilderFlag.FP16)
    config.flags = config.flags & ~( 1 << int(trt.BuilderFlag.TF32) )
    config.max_workspace_size =  15 * (1 << 30)
    #config.profiling_verbosity = True
    if useTimeCache:
        cache = config.create_timing_cache(timeCache)
        config.set_timing_cache(cache, False)

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing ONNX file!")

    #print_net(network)
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    make_logsoftmax_fp32(network)

    for i in range(5):
        inputTensor = network.get_input(i)
        name = inputTensor.name
        if name == "encoder_out":
            min_shape = (1, 16, 256)
            max_shape = (16, 256, 256)
            #opt_shape = (4, 64, 256)
            opt_shape = max_shape
            #opt_shape = min_shape
        elif name == "encoder_out_lens":
            min_shape = (1,)
            max_shape = (16,)
            #opt_shape = (4,)
            opt_shape = max_shape
            #opt_shape = min_shape
        elif name == "hyps_pad_sos_eos":
            min_shape = (1, 10, 64)
            max_shape = (16, 10, 64)
            #opt_shape = (4, 10, 64)
            opt_shape = max_shape
            #opt_shape = min_shape
        elif name == "hyps_lens_sos":
            min_shape = (1, 10,)
            max_shape = (16, 10,)
            #opt_shape = (4, 10,)
            opt_shape = max_shape
            #opt_shape = min_shape
        elif name == "ctc_score":
            min_shape = (1, 10,)
            max_shape = (16, 10,)
            #opt_shape = (4, 10,)
            opt_shape = max_shape
            #opt_shape = min_shape
        profile.set_shape(inputTensor.name, min_shape, opt_shape, max_shape)
        
    config.add_optimization_profile(profile)

    #network.unmark_output(network.get_output(0))  
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")

    if useTimeCache and not os.path.isfile(timeCacheFile):
        timeCache = config.get_timing_cache()
        timeCacheString = timeCache.serialize()
        with open(timeCacheFile, 'wb') as f:
            f.write(timeCacheString)
            print("Succeeded saving .cache file!")

    with open(trtFile, 'wb') as f:
        f.write(engineString)

