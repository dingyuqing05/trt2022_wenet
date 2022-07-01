import os
import sys
import argparse
import numpy as np
#import torch
#from transformers import T5ForConditionalGeneration

model_file = '/target/python/enc/encoder.npy'
saved_dir  = '/target/python/enc/bin_model'


if __name__ == "__main__":
    os.system("mkdir -p "+ saved_dir)

    ews = np.load(model_file, allow_pickle='TRUE')
    ews = ews.item()

    for name in ews:
        saved_path = saved_dir + "/" + name + ".bin"
        cur = ews[name]
        if name.endswith(".weight") and len(cur.shape)==2:
            cur = cur.transpose((1,0))
            #print(name, cur.shape)
        if name.endswith(".pointwise_conv1.weight"):
            cur = cur.transpose((1,0,2))
            #print(name, cur.shape)

        if name.endswith(".pointwise_conv2.weight"):
            cur = cur.transpose((1,0,2))
            #print(name, cur.shape)

        if name.endswith(".depthwise_conv.weight") and len(cur.shape)==3:
            cur = cur.transpose((2,1,0))
            #print(name, cur.shape)

        cur.tofile(saved_path)


    print("extract Wenet encoder model weight finish!")

