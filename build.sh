

<< COM
pushd /target/csrc
bash compile.sh
popd
COM

#<< COM
pushd /target/trt_ft_wenet
bash compile.sh
popd

#<< COM
pushd /target/python/enc
python onnx2torch.py
python /target/trt_ft_wenet/src/fastertransformer/models/wenet/ExtractEncoderToBIN.py
python opt_onnx.py
python enc2trt.py
popd
#COM

#<< COM
pushd /target/python/dec
python onnx2torch.py
python /target/trt_ft_wenet/src/fastertransformer/models/wenet/ExtractDecoderToBIN.py
python opt_onnx.py
python dec2trt.py
popd
#COM
