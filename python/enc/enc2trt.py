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


#onnxFile = "/workspace/encoder.onnx"
onnxFile = "/target/python/enc/encoder_opt.onnx"
trtFile = "/target/encoder.plan"
timeCacheFile = "/target/python/enc/tc_fp16_ln16.cache"
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
logger = trt.Logger(trt.Logger.INFO)
#logger = trt.Logger(trt.Logger.VERBOSE)
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
    config.max_workspace_size = 15 << 30
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

    inputTensor = network.get_input(0)
    inputTensor2 = network.get_input(1)
    #profile.set_shape(inputTensor.name, (1, 16, 80), (4, 64, 80), (16, 256, 80))
    #profile.set_shape(inputTensor2.name, (1,), (4,), (16,))
    profile.set_shape(inputTensor.name, (1, 16, 80), (16, 256, 80), (16, 256, 80))
    profile.set_shape(inputTensor2.name, (1,), (16,), (16,))
    #profile.set_shape(inputTensor.name, (1, 16, 80), (1, 16, 80), (16, 256, 80))
    #profile.set_shape(inputTensor2.name, (1,), (1,), (16,))
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
