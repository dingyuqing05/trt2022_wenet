#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 dingyuqing
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import os


from torch import nn
from torch.nn import functional as F

#import pytorch_quantization.tensor_quant as tensor_quant
from pytorch_quantization import tensor_quant
#import pytorch_quantization.nn.modules._utils as _utils
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
#import pytorch_quantization.nn as quant_nn


USE_QUANT = os.getenv('WENET_INT8_QAT')
if USE_QUANT == 'True':
    USE_QUANT = True
elif USE_QUANT == 'False':
    USE_QUANT = False
else:
    assert False

#QUANT_DESC_8BIT_PER_TENSOR_LEARN = tensor_quant.QuantDescriptor(num_bits=8, amax=1., learn_amax=True)
QUANT_DESC_8BIT_PER_TENSOR_R = tensor_quant.QuantDescriptor(num_bits=8, fake_quant=False)

class QuantLinear(nn.Linear):
    #default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    #default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    def __init__(self, in_features, out_features, bias=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.input_quantizer = TensorQuantizer(tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
        self.weight_quantizer = TensorQuantizer(tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
        self.out_quantizer = TensorQuantizer(tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
        """
        self.input_quantizer = TensorQuantizer(QUANT_DESC_8BIT_PER_TENSOR_R)
        self.weight_quantizer = TensorQuantizer(QUANT_DESC_8BIT_PER_TENSOR_R)
        self.out_quantizer = TensorQuantizer(QUANT_DESC_8BIT_PER_TENSOR_R)
        """

    def forward(self, input):
        quant_input = self.input_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)

        output = F.linear(quant_input, quant_weight, bias=None)
        quant_output = self.out_quantizer(output)

        output = quant_output + self.bias

        return output

def disable_quant(model):
    for layer in model.modules():
        if isinstance(layer, QuantLinear):
            layer.input_quantizer.disable()
            layer.weight_quantizer.disable()
            layer.out_quantizer.disable()

def enable_calib(model):
    for layer in model.modules():
        if isinstance(layer, QuantLinear):
            layer.input_quantizer.enable()
            layer.weight_quantizer.enable()
            layer.out_quantizer.enable()

            layer.input_quantizer.enable_calib()
            #layer.weight_quantizer.enable_calib()
            layer.out_quantizer.enable_calib()

            layer.input_quantizer.disable_quant()
            layer.weight_quantizer.disable_quant()
            layer.out_quantizer.disable_quant()


def load_calib_amax(model):
    for layer in model.modules():
        if isinstance(layer, QuantLinear):

            layer.input_quantizer.load_calib_amax()
            layer.out_quantizer.load_calib_amax()

            layer.input_quantizer.disable_calib()
            #layer.weight_quantizer.disable_calib()
            layer.out_quantizer.disable_calib()

            layer.input_quantizer.enable_quant()
            layer.weight_quantizer.enable_quant()
            layer.out_quantizer.enable_quant()


