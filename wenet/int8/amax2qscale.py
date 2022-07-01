import os
import sys
import numpy as np
import argparse
import torch
from collections import OrderedDict



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_onnx_dir', required=True, help='checkpoint model')
    parser.add_argument('--enc_layer_num', default=12, type=int, required=False)
    parser.add_argument('--dec_layer_num', default=6, type=int, required=False)
    args = parser.parse_args()

    enc_layer_num = args.enc_layer_num
    dec_layer_num = args.dec_layer_num

    sd = torch.load(args.checkpoint)

    new_sd = OrderedDict()
    amax_sd = dict()

    for wname in sd:
        if wname.endswith("_amax"):
            #print(wname)
            amax_sd[wname] = sd[wname]
        else:
            new_sd[wname] = sd[wname]

    if len(amax_sd) > 0:
        for i in range(enc_layer_num):
            wname = 'encoder.encoders.' + str(i) + '.feed_forward.w_1.weight'
            w_amax = sd[wname].amax()
            wname = wname + '._amax'
            amax_sd[wname] = w_amax

            wname = 'encoder.encoders.' + str(i) + '.feed_forward.w_2.weight'
            w_amax = sd[wname].amax()
            wname = wname + '._amax'
            amax_sd[wname] = w_amax

            wname = 'encoder.encoders.' + str(i) + '.feed_forward_macaron.w_1.weight'
            w_amax = sd[wname].amax()
            wname = wname + '._amax'
            amax_sd[wname] = w_amax

            wname = 'encoder.encoders.' + str(i) + '.feed_forward_macaron.w_2.weight'
            w_amax = sd[wname].amax()
            wname = wname + '._amax'
            amax_sd[wname] = w_amax

        for i in range(dec_layer_num):
            wname = 'decoder.decoders.' + str(i) + '.feed_forward.w_1.weight'
            w_amax = sd[wname].amax()
            wname = wname + '._amax'
            amax_sd[wname] = w_amax

            wname = 'decoder.decoders.' + str(i) + '.feed_forward.w_2.weight'
            w_amax = sd[wname].amax()
            wname = wname + '._amax'
            amax_sd[wname] = w_amax

        for w in amax_sd:
            amax_sd[w] = amax_sd[w].cpu().numpy()

        torch.save(new_sd, args.checkpoint)
        np.save(args.output_onnx_dir + "/wenet_amax.npy", amax_sd)


