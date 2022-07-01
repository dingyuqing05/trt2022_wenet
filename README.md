## 总述

本项目为 [WeNet](https://github.com/wenet-e2e/wenet) 在 TensorRT 上的推理加速优化方案，项目对 WeNet 中的 Encoder 模型和 Decoder 模型做了深度的性能优化，并且支持了FP32, FP16精度的混合计算，最终相比于baseline达到了一个非常高的加速比。

#### 优化技术

- FP16精度计算，支持encoder和decoder
- 算子融合，比如LayerNorm(导出onnx后被替换成了10多个小算子）, Bias+Residual+LayerNorm
- cache机制，缓存部分重复计算的结果，比如pos_emb，attention_mask
- 特殊kernel的优化，比如depthwise_convolution，log_softmax
- cudagraph，减少kernel lauch的时间
- ......

#### 步骤说明

镜像是初赛提供的镜像，参考：https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/hackathon/setup.md

###### 初赛结果复现

```bash
cd /
git clone https://github.com/dingyuqing05/trt2022_wenet.git
rm -rf target
mv trt2022_wenet target
cd /workspace
bash buildFromWorkspace.sh
```

###### 复赛结果复现

配置环境

```bash
cd /target
bash env.sh
```

采用了aishell数据集, 可在官网下载：http://www.aishelltech.com/kysjcp

```bash
mkdir -p /mnt/data/asr-data/OpenSLR/33
cd /mnt/data/asr-data/OpenSLR/33
```

将下载后的数
data_aishell.tgz，
resource_aishell.tgz
放到 /mnt/data/asr-data/OpenSLR/33 目录下：

```bash
/mnt/data/asr-data/OpenSLR/33/data_aishell.tgz
/mnt/data/asr-data/OpenSLR/33/resource_aishell.tgz
```

训练，模型采用了和初赛一样的结构，详细配置见
wenet/examples/aishell/s0/ conf/dyq_conformer.yaml
因为算力有限，这里我只采用了前10%的aishell的数据做训练和测试，注意数据集大小会影响vocab_size参数的大小。

```bash
cd /target
bash train_int8qat.sh
```

如果训练出现错误：

```bash
RuntimeError: DataLoader worker (pid 321390) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
```

需要修改docker容器的shm大小, 参考：https://blog.csdn.net/wd18508423052/article/details/116306096
训练完成后，导出onnx模型，转成TensorRT FP16模型并测试torch速度和TensorRT速度，验证结果：

```bash
cd /target
bash fp16_2trt_test.sh
```


## 原始模型

### 模型简介

[WeNet](https://github.com/wenet-e2e/wenet) 是一款面向工业落地应用的语音识别工具包，提供了从语音识别模型的训练到部署的一条龙服务，其具体的用途和效果可以参考官方github上的介绍：https://github.com/wenet-e2e/wenet ,

目前为止其在github上收获了2k+Star，说明其被众多的相关从业者所青睐。由其[PR文章](https://www.infoq.cn/article/oqlNys5qlQWRkYuEZkEG)可知，目前 WeNet 应用到了喜马拉雅、作业帮、京东、腾讯等数百家公司，他们采用 WeNet 构建自己语音服务，覆盖智能车载、智能家居、智能客服、音频内容生产、直播、会议等语音识别应用场景。

- 模型Encoder由多个ConformerBlock构成，具体结构可以参考论文：https://arxiv.org/pdf/2005.08100v1.pdf ，其中比较有特色的是 convolution module 结构
- 模型Decoder由标准的Decoder Block构成

### 模型优化的难点

模型可以通过官方提供的代码导出到ONNX，但ONNX转TensorRT时，存在掉精度，性能不够好的问题。具体有：

- FP16时模型精度掉的严重，模型中Reduce相关的算子可能存在数据溢出的问题
- LayerNorm算子被打散成10多个小算子，而TensorRT对这些小算子的fusion不够好，导致掉精度以及速度不够快
- 导出的ONNX模型中存在重复计算的部分，比如生成mask, pos_emb, 导致多余的重复计算
- TensorRT对部分算子的融合不够好，比如 bias_residual_layernorm, masked_softmax, multi-head attention
- TessorRT对部分算子的实现不够快，比如 depthwise_convolution, logsoftmax
- INT8 PTQ影响精度
- 存在大量细碎的kernel，导致 kernel lauch 耗时占比较高
- ......

这些问题导致该模型要达到极致的性能需要做很多深度的工程优化，存在较大的开发工作量和较大的难度，有一定的挑战性。

## 优化过程

- 初赛只提供了onnx模型，最直接的尝试是参考cookbook中提供的TensorRT转换例子直接转到成TensorRT模型。
- 针对encoder模型，直接转换时会报Slice的数据类型错误，netron分析onnx图后发现Not节点的输出直接送进了Slice节点，而TensorRT的Slice不支持bool输入，于是在两个节点间插入Cast节点做一个类型转换，解决这个问题。
- encoder直接转到TensorRT后，FP16结果diff很大，于是尝试FP32，结果diff较小，猜测是模型中存在数据溢出。
- github上找到wenet的pytorch代码，大致阅读模型结构后发现里面有很多LayerNorm算子，按照经验LayerNorm可能会出现数据溢出。测试后发现LayerNorm算子确实被打散成了10多个小算子，而这些小算子中的reduce操作容易造成数据溢出。另一方面LayerNorm打散成小算子后，速度也会变慢。为了解决这两个问题，实现了LayerNorm插件。提速的同时也解决了encoder diff大的问题。
- decoder模型中也存在LayerNorm算子，可以复用LayerNorm插件，复用后模型可以直接转到TensorRT下。
- 优化到此后score可以到1890分
- 更进一步的优化可以采用类似的思路在onnx图上修修改改，合并小算子，最开始也做过一些尝试，比如masked_softmax插件，但做下来感觉这种思路在优化上还是不够彻底。
- 另一种思路是手工搭建网络，但自动转换生成的onnx网路图可读性太差。最好是有onnx模型对应的原始pytorch代码作参考和对齐。分析onnx模型中的权重名称和shape后，得到网络结构的一些基本参数，比如layer_num, hidden_size, head_num等，同时和wenet-pytorch代码作对比，反复对比，并试验后成功在wenet-pytorch下还原出了模型，并和onnx模型对齐了结果。
- 梳理出wenet-pytorch中的模型网络结构，按照经验画出算子融合后的网络结构。
- 采用FasterTransformer实现出了encoder部分，并做成TensorRT插件。
- 采用trtexec做性能的profile，发现conv_module中的depthwise_convolution耗时百分比高，这里针对该算子实现了定制化的cuda kernel 。
- 优化到此后score可以到4000分
- 采用FasterTransformer实现出decoder部分，并做成TensorRT插件。
- 优化到此后score可以到8000分
- 尝试过做变长输入的计算，以避免padding带来的无效计算，但发现评估脚本中包含了padding部分的结果比对，导致这项技术无法发挥大作用。
- 分析模型结构后，发现pos_emb部分的结果是可以提前计算好并缓存起来的，从而避免重复计算。
- 发现评测脚本中encoder模型的几个输出没有参与计算，所以可以剪掉，避免不必要的计算。
- 用trtexec对decoder profile后觉得log_softmax有优化空间，同时猜测该部分可能会影响精度，所以定制化地实现了该kernel，确实对精度和速度有改善。
- nsight systems 分析后发现存在不少kernel lanch的开销，所以添加了cudagraph的支持。截图见根目录下的 without_cudagraph.png  和 with_cudagraph.png
- 优化到此后score可以到10000分
- 为了更进一步的速度，尝试 INT8 QAT, 这里只对encoder block中的两个ffn模块做int8计算。确定好需要量化的Tensor后，修改wenet的pytorch代码，添加QAT量化支持，训练的具体loss变化可以看日志：wenet/examples/aishell/s0/train.log   中查看。目前只完成了int8qat训练，推理部分的int8qat目前只开发了一半，还未完成。最终训练部分的loss：
  
  INT8-QAT train loss

| Models           | PyTorch     | PyTorch + INT8 QAT |
| -------------------| ---------------- | ----------------------- |
| Encoder (INT8QAT）+ Decoder(FP16)        | 32.97482916332173           | 31.75881217961867                   |



## 精度与加速效果

本地测试采用的镜像是初赛提供的镜像，参考：https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/hackathon/setup.md
测试机器是自用的游戏机：RTX2070 + AMD Ryzen 7 1700 + 64G Memory

Baseline采用pytorch1.11运行, 模型采用自己训练的模型（见复现步骤），测试数据采用的是初赛的测试数据。

PyTorch(FP32)+Encoder

```bash
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

   1,  16,  16.152,9.906e+02
   1, 256,  18.377,1.393e+04
   1,  64,  17.294,3.701e+03
  16,  16,  18.070,1.417e+04
  16, 256,  21.513,1.904e+05
  16,  64,  17.931,5.711e+04
   4,  16,  17.329,3.693e+03
   4, 256,  18.349,5.581e+04
   4,  64,  18.687,1.370e+04
avg at: 1.819e+01, [16.151677799999998, 18.377048966666667, 17.29381833333333, 18.069583966666663, 21.513327999999998, 17.930813, 17.329331733333333, 18.349227999999997, 18.687338366666665]
avg tp: 3.928e+04,
```

PyTorch(FP32)+Decoder

```bash
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

   1,  16,   7.838,2.041e+03
   1, 256,   8.026,3.189e+04
   1,  64,   7.940,8.060e+03
  16,  16,  46.974,5.450e+03
  16, 256,  73.328,5.586e+04
  16,  64,  52.282,1.959e+04
   4,  16,  12.970,4.934e+03
   4, 256,  19.694,5.199e+04
   4,  64,  14.328,1.787e+04
avg at: 2.704e+01, [7.837684733333333, 8.026468366666666, 7.940481166666667, 46.97404053333333, 73.3283487, 52.28224856666667, 12.970208333333334, 19.694452266666666, 14.327813866666668]
avg tp: 2.197e+04,
```

TensorRT(FP16)+Encoder

```bash
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

   1,  16,   1.585,1.010e+04,9.431e-03,1.151e-03,0.000e+00,0.000e+00, Good
   1, 256,   1.697,1.509e+05,1.044e-02,1.078e-03,0.000e+00,0.000e+00, Good
   1,  64,   1.526,4.194e+04,4.111e-03,1.138e-03,0.000e+00,0.000e+00, Good
  16,  16,   1.549,1.653e+05,8.490e-03,1.132e-03,0.000e+00,0.000e+00, Good
  16, 256,   6.363,6.437e+05,1.275e-02,1.090e-03,0.000e+00,0.000e+00, Good
  16,  64,   2.413,4.243e+05,1.006e-02,1.041e-03,0.000e+00,0.000e+00, Good
   4,  16,   1.523,4.203e+04,8.490e-03,1.175e-03,0.000e+00,0.000e+00, Good
   4, 256,   2.629,3.895e+05,1.215e-02,1.243e-03,0.000e+00,0.000e+00, Good
   4,  64,   1.651,1.550e+05,1.369e-02,1.136e-03,0.000e+00,0.000e+00, Good
avg a0: 9.956e-03, target: 3.500e-02
avg r0: 1.132e-03, target: 2.000e-03
avg at: 2.326e+00, [1.5846379000000002, 1.6968412333333334, 1.5259125666666666, 1.5486122666666666, 6.3630618, 2.4132516, 1.5228222666666664, 2.6292861, 1.6512201333333332]
avg tp: 2.248e+05,
```

TensorRT(FP16)+Decoder

```bash
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

   1,  16,   2.011,7.957e+03,2.280e-02,2.015e-04,0.000e+00,0.000e+00, Good
   1, 256,   2.558,1.001e+05,2.728e-02,2.131e-04,0.000e+00,0.000e+00, Good
   1,  64,   2.128,3.008e+04,2.540e-02,2.074e-04,0.000e+00,0.000e+00, Good
  16,  16,  17.266,1.483e+04,3.356e-02,2.045e-04,0.000e+00,0.000e+00, Good
  16, 256,  22.680,1.806e+05,3.172e-02,2.201e-04,0.000e+00,0.000e+00, Good
  16,  64,  18.141,5.645e+04,2.944e-02,2.147e-04,0.000e+00,0.000e+00, Good
   4,  16,   5.171,1.238e+04,2.987e-02,2.006e-04,0.000e+00,0.000e+00, Good
   4, 256,   6.640,1.542e+05,2.770e-02,2.198e-04,0.000e+00,0.000e+00, Good
   4,  64,   5.432,4.712e+04,2.960e-02,2.183e-04,0.000e+00,0.000e+00, Good
avg a0: 2.860e-02, target: 4.000e-01
avg r0: 2.111e-04, target: 3.000e-04
avg tp: 6.708e+04,
```

## 经验与体会

第一次参加这种模型加速比赛，非常感谢NVIDIA提供了这样一个活动让相关方向的人有这样一个交流的机会。一方面认识了不少有趣的人，另一方面也学到了很多以前没有用过的技能：

- 第一次使用graph-surgeon，之前改onnx图都是直接用onnx自带的api
- 第一次使用cudagraph，之前一直没有合适的场景或者时间去用它
- 第一次使用FasterTransformer5.0并集成到TensorRT，之前只是看过FasterTransformer4.0的代码
- 第一次由onnx模型去还原torch模型
- 第一次训练wenet模型，也是第一次使用pytorch-quantization工具做INT8-QAT，之前自己更多的是聚焦在推理引擎侧
- ......

总之，这是一段非常有趣且有收获的经历。再次感谢NVIDIA提供这样的活动，也非常感谢DevTech团队的老师们的辛苦付出与活动过程中的各种帮助，期待将来会有更多相关的活动举办，让大家有更多互相交流互相学习的机会。
