#!/bin/bash
set -e

export PATH=$PATH:/root/anaconda3/bin

pushd /target

__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate torch1.11

#<< COM
pushd /target/wenet/examples/aishell/s0
bash run.sh --stage -1 --stop-stage -1
bash run.sh --stage 0 --stop-stage 0
mv data data_bk
python gen_data.py
bash run.sh --stage 1 --stop-stage 3
bash run.sh --stage 4 --stop-stage 4
popd
#COM

popd



