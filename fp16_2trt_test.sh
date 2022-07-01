#!/bin/bash
set -e

#<< COM
pushd /target/trt_ft_wenet
bash compile.sh
popd
#COM


#<< COM
pushd /target/wenet/int8
bash fp16_export.sh
popd
#COM

#<< COM
pushd /target/python/enc
python fp16_2torch.py
python /target/trt_ft_wenet/src/fastertransformer/models/wenet/ExtractEncoderToBIN.py
python fp16_opt.py
python enc2trt.py
popd
#COM

#<< COM
pushd /target/python/dec
python fp16_2torch.py
python /target/trt_ft_wenet/src/fastertransformer/models/wenet/ExtractDecoderToBIN.py
python fp16_opt.py
python dec2trt.py
popd
#COM


#<< COM
pushd /target/wenet/int8
bash fp16_test.sh
python trt_test.py
#COM
