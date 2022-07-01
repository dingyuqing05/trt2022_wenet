#!/bin/bash
set -e

export PATH=$PATH:/root/anaconda3/bin

pushd /target

#<< COM
wget -c https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
chmod +x Anaconda3-2022.05-Linux-x86_64.sh
./Anaconda3-2022.05-Linux-x86_64.sh -b
#conda init
conda create --name torch1.11 python=3.8.10 -y
#COM

#<< COM
wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
tar -zxvf cmake-3.20.0.tar.gz
cd cmake-3.20.0
cp ../bootstrap ./
./bootstrap
make -j16
make install
apt install ninja-build
#COM



__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate torch1.11

#<< COM
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu115
pip3 install onnxruntime-gpu
#COM




#<< COM
git clone https://github.com/pytorch/audio.git
mv audio torchaudio

pushd torchaudio
git checkout release/0.11
export CUDACXX=/usr/local/cuda/bin/nvcc
python setup.py install
popd
#COM

#<< COM
git clone https://github.com/NVIDIA/TensorRT.git
pushd TensorRT/tools/pytorch-quantization
git checkout release/8.2
python setup.py install
popd
#COM

#<< COM
pushd /target/wenet
pip install -r requirements.txt
popd
#COM



