
export TRT_LIBPATH=/usr/lib/x86_64-linux-gnu

CUR_DIR=`pwd`

rm -rf build
mkdir -p build
cd build
cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE=OFF \
	-DCMAKE_INSTALL_PREFIX=${CUR_DIR}/install \
	-DBUILD_TF=OFF \
	-DBUILD_PYT=OFF \
	-DBUILD_TRT=OFF \
	-DBUILD_MULTI_GPU=OFF \
	-DUSE_NVTX=OFF \
	-DBUILD_EXAMPLE=OFF \
	-DBUILD_TEST=OFF \
	-DBUILD_TRT=ON \
	-DBUILD_ORGIN_NET=OFF \
	..


make -j$(nproc) 

cp ${CUR_DIR}/build/lib/libtrt_wenet.so /target/libtrt_wenet.so
#make -j$(nproc) install

#cp ${CUR_DIR}/out/libtrt2022_plugin.so.SOVERSION /target/libtrt2022_plugin.so

