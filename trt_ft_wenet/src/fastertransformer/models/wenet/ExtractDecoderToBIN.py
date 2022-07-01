import os
import sys
import argparse
import numpy as np
#import torch
#from transformers import T5ForConditionalGeneration

model_file = '/target/python/dec/decoder.npy'
saved_dir  = '/target/python/dec/bin_model'


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

        cur.tofile(saved_path)


    print("extract Wenet decoder model weight finish!")

