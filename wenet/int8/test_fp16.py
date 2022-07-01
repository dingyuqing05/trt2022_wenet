# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import torch
import yaml
import logging

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.mask import make_pad_mask

try:
    import onnxruntime
except ImportError:
    print('Please install onnxruntime-gpu!')
    sys.exit(1)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def reg_custom_op():
    import torch.onnx.symbolic_registry as sr
    def layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enabled):
        type_id = 1 # 0->FP32, 1->FP16
        return g.op("CustomLayerNormPluginDynamic", input, gamma_f = weight.node()['value'].numpy(), beta_f = bias.node()['value'].numpy(), ld_i = 256, type_id_i = type_id)
    ver_list = [9, 10, 11, 12, 13]
    for ver in ver_list:
        #sr.register_op("layer_norm", layer_norm, '', ver)
        pass

class Encoder(torch.nn.Module):
    def __init__(self,
                 encoder: BaseEncoder,
                 ctc: CTC,
                 beam_size: int = 10):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.beam_size = beam_size

    def forward(self, speech: torch.Tensor,
                speech_lengths: torch.Tensor,):
        """Encoder
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        Returns:
            encoder_out: B x T x F
            encoder_out_lens: B
            ctc_log_probs: B x T x V
            beam_log_probs: B x T x beam_size
            beam_log_probs_idx: B x T x beam_size
        """
        encoder_out, encoder_mask = self.encoder(speech,
                                                 speech_lengths,
                                                 -1, -1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_log_probs = self.ctc.log_softmax(encoder_out)
        encoder_out_lens = encoder_out_lens.int()
        beam_log_probs, beam_log_probs_idx = torch.topk(
            ctc_log_probs, self.beam_size, dim=2)
        return encoder_out, encoder_out_lens, ctc_log_probs, \
            beam_log_probs, beam_log_probs_idx


class Decoder(torch.nn.Module):
    def __init__(self,
                 decoder: TransformerDecoder,
                 ctc_weight: float = 0.5,
                 reverse_weight: float = 0.0,
                 beam_size: int = 10):
        super().__init__()
        self.decoder = decoder
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.beam_size = beam_size

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_lens: torch.Tensor,
                hyps_pad_sos_eos: torch.Tensor,
                hyps_lens_sos: torch.Tensor,
                #r_hyps_pad_sos_eos: torch.Tensor,
                ctc_score: torch.Tensor):
        """Encoder
        Args:
            encoder_out: B x T x F
            encoder_lens: B
            hyps_pad_sos_eos: B x beam x (T2+1),
                        hyps with sos & eos and padded by ignore id
            hyps_lens_sos: B x beam, length for each hyp with sos
            ctc_score: B x beam, ctc score for each hyp
        Returns:
            decoder_out: B x beam x T2 x V
            best_index: B
        """
        B, T, F = encoder_out.shape
        bz = self.beam_size
        B2 = B * bz
        encoder_out = encoder_out.repeat(1, bz, 1).view(B2, T, F)
        encoder_mask = ~make_pad_mask(encoder_lens, T).unsqueeze(1)
        encoder_mask = encoder_mask.repeat(1, bz, 1).view(B2, 1, T)
        T2 = hyps_pad_sos_eos.shape[2] - 1
        hyps_pad = hyps_pad_sos_eos.view(B2, T2 + 1)
        hyps_lens = hyps_lens_sos.view(B2,)
        hyps_pad_sos = hyps_pad[:, :-1].contiguous()
        hyps_pad_eos = hyps_pad[:, 1:].contiguous()
        
        r_hyps_pad_sos = None

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad_sos, hyps_lens, r_hyps_pad_sos,
            self.reverse_weight)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        V = decoder_out.shape[-1]
        decoder_out = decoder_out.view(B2, T2, V)
        mask = ~make_pad_mask(hyps_lens, T2)  # B2 x T2

        # mask index, remove ignore id
        index = torch.unsqueeze(hyps_pad_eos * mask, 2)
        score = decoder_out.gather(2, index).squeeze(2)  # B2 X T2
        # mask padded part
        score = score * mask
        decoder_out = decoder_out.view(B, bz, T2, V)

        score = torch.sum(score, axis=1)  # B2
        score = torch.reshape(score, (B, bz)) + self.ctc_weight * ctc_score
        best_index = torch.argmax(score, dim=1)
        return decoder_out, best_index

from glob import glob
from time import time_ns

out_dir = '/target/wenet/int8/torch_out/'
os.system("mkdir " + out_dir)
dataFilePath = "/workspace/data/"
tableHead = \
"""
bs: Batch Size
sl: Sequence Length
lt: Latency (ms)
tp: throughput (word/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
a1: maximum of absolute difference of output 1
r1: median of relative difference of output 1
----+----+--------+---------+---------+---------+---------+---------+-------------
  bs|  sl|      lt|       tp|       a0|       r0|       a1|       r1| output check
----+----+--------+---------+---------+---------+---------+---------+-------------
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--reverse_weight', default=-1.0, type=float,
                        required=False,
                        help='reverse weight for bitransformer,' +
                        'default value is in config file')
    parser.add_argument('--ctc_weight', default=-1.0, type=float,
                        required=False,
                        help='ctc weight, default value is in config file')
    parser.add_argument('--beam_size', default=10, type=int, required=False,
                        help="beam size would be ctc output size")
    parser.add_argument('--output_onnx_dir',
                        default="onnx_model",
                        help='output onnx encoder and decoder directory')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')
    args = parser.parse_args()

    #reg_custom_op()
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if args.reverse_weight != -1.0 and 'reverse_weight' in configs['model_conf']:
        configs['model_conf']['reverse_weight'] = args.reverse_weight
        print("Update reverse weight to", args.reverse_weight)
    if args.ctc_weight != -1:
        print("Update ctc weight to ", args.ctc_weight)
        configs['model_conf']['ctc_weight'] = args.ctc_weight
    configs["encoder_conf"]["use_dynamic_chunk"] = False

    if False:
        configs['cmvn_file'] = None
        configs["input_dim"] = 80
        configs["output_dim"] = 4233
        wdir = "/target/python/"
        ews = np.load(wdir + "/enc/encoder.npy", allow_pickle='TRUE').item()
        #dws = np.load(wdir + "/dec/decoder.npy", allow_pickle='TRUE')
        cmvn_mean = ews["encoder.global_cmvn.mean"]
        cmvn_istd = ews["encoder.global_cmvn.istd"]
    else:
        #configs["input_dim"] = 80
        #configs["output_dim"] = 4233
        cmvn_mean = None
        cmvn_istd = None


    model = init_asr_model(configs, mean=cmvn_mean, istd=cmvn_istd)
    load_checkpoint(model, args.checkpoint)
    model.eval().cuda()

    if True:
        encoder = Encoder(model.encoder, model.ctc, args.beam_size)
        encoder.eval()

        #-------------------------------------------------------------------------------
        print("Test Encoder Part!")
        with torch.no_grad():
            if True:
                print(tableHead)  # for standard output
                at_list = list()
                tp_list = list()
                ccc = 0
                for ioFile in sorted(glob(dataFilePath + "./encoder-*.npz")):
                    ioData = np.load(ioFile)
                    speech = ioData['speech']
                    speech_lengths = ioData['speech_lengths']
                    batchSize, sequenceLength, _ = speech.shape
                    if batchSize > 16 or sequenceLength > 1024:
                        continue
                    np_speech = speech
                    np_speech_lengths = speech_lengths

                    speech = torch.from_numpy(np_speech).cuda()
                    speech_lens = torch.from_numpy(np_speech_lengths).cuda()

                    # warm up
                    for i in range(10):
                        o0, o1, o2, o3, o4 = encoder(speech, speech_lens)
                    t0 = time_ns()
                    for i in range(30):
                        o0, o1, o2, o3, o4 = encoder(speech, speech_lens)
                    t1 = time_ns()
                    timePerInference = (t1-t0)/1000/1000/30

                    #print(ioData['encoder_out_lens'])
                   
                    if True:
                        i0 = np_speech
                        i1 = np_speech_lengths
                        o0 = o0.detach().cpu().numpy()
                        o1 = o1.detach().cpu().numpy()
                        out_file = out_dir + os.path.basename(ioFile)
                        np.savez(out_file, speech=i0, speech_lengths=i1, encoder_out=o0, encoder_out_lens=o1)

                    string = "%4d,%4d,%8.3f,%9.3e"%(batchSize,
                                                    sequenceLength,
                                                    timePerInference,
                                                    batchSize*sequenceLength/timePerInference*1000)
                    print(string)
                    at_list.append(timePerInference)
                    tp_list.append(batchSize*sequenceLength/timePerInference*1000)

                string = "avg at: %9.3e,"%(sum(at_list)/len(at_list))
                print(string, at_list)
                string = "avg tp: %9.3e,"%(sum(tp_list)/len(tp_list))
                print(string)

    if True:
        decoder = Decoder(model.decoder, model.ctc_weight, model.reverse_weight, args.beam_size)
        decoder.eval()
        #-------------------------------------------------------------------------------
        print("Test Decoder Part!")
        with torch.no_grad():
            print(tableHead)  # for standard output
            at_list = list()
            tp_list = list()
            for ioFile in sorted(glob(dataFilePath + "./decoder-*.npz")):
                ioData = np.load(ioFile)
                encoder_out_np = ioData['encoder_out']
                encoder_out_lens_np = ioData['encoder_out_lens']
                hyps_pad_sos_eos_np = ioData['hyps_pad_sos_eos']
                hyps_lens_sos_np = ioData['hyps_lens_sos']
                ctc_score_np = ioData['ctc_score']
                batchSize, sequenceLength, _ = encoder_out_np.shape
                if batchSize > 16 or sequenceLength > 256:
                    continue
               
                encoder_out = torch.from_numpy(encoder_out_np).cuda()
                encoder_out_lens = torch.from_numpy(encoder_out_lens_np).cuda()
                hyps_pad_sos_eos = torch.from_numpy(hyps_pad_sos_eos_np).cuda()
                hyps_lens_sos = torch.from_numpy(hyps_lens_sos_np).cuda()
                ctc_score = torch.from_numpy(ctc_score_np).cuda()

                # warm up
                for i in range(10):
                    o0, o1 = decoder(encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, ctc_score)
                # test infernece time
                t0 = time_ns()
                for i in range(30):
                    o0, o1 = decoder(encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, ctc_score)
                t1 = time_ns()
                timePerInference = (t1-t0)/1000/1000/30

                if True:
                    o0 = o0.detach().cpu().numpy()
                    o1 = o1.detach().cpu().numpy()
                    out_file = out_dir + os.path.basename(ioFile)
                    np.savez(out_file, 
                            encoder_out=encoder_out_np, 
                            encoder_out_lens=encoder_out_lens_np, 
                            hyps_pad_sos_eos=hyps_pad_sos_eos_np, 
                            hyps_lens_sos=hyps_lens_sos_np, 
                            ctc_score=ctc_score_np, 
                            decoder_out=o0, best_index=o1)

                string = "%4d,%4d,%8.3f,%9.3e"%(batchSize,
                                                sequenceLength,
                                                timePerInference,
                                                batchSize*sequenceLength/timePerInference*1000)
                print(string)
                at_list.append(timePerInference)
                tp_list.append(batchSize*sequenceLength/timePerInference*1000)

            string = "avg at: %9.3e,"%(sum(at_list)/len(at_list))
            print(string, at_list)
            string = "avg tp: %9.3e,"%(sum(tp_list)/len(tp_list))
            print(string)


