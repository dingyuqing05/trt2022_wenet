#!/bin/bash

export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$PYTHONPATH:/target/wenet
export WENET_INT8_QAT=False

model_dir=/target/wenet/int8
weight_file=/target/wenet/examples/aishell/s0/exp/conformer/100.pt
onnx_model_dir=/target/python

cp $weight_file model.pt

__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate torch1.11

python3 export_onnx_gpu.py \
	--config=$model_dir/train_conformer.yaml \
	--checkpoint=model.pt \
	--ctc_weight=0.5 \
	--output_onnx_dir=$onnx_model_dir \

	#--fp16
