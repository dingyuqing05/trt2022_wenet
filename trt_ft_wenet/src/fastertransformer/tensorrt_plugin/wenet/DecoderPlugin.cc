/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DecoderPlugin.h"

using namespace fastertransformer;

namespace nvinfer1 {

// class WenetDecoderPlugin ---------------------------------------------------------------------------
WenetDecoderPlugin::WenetDecoderPlugin(const std::string& name,
                                       size_t max_batch_size,
                                       size_t max_seq_len,
                                       size_t head_num,
                                       size_t size_per_head,
                                       size_t inter_size,
                                       size_t d_model,
                                       size_t num_layer,
                                       size_t vocab_size,
                                       int sm,
                                       float q_scaling,
                                       int useFP16):
    name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.max_batch_size = max_batch_size;
    m_.max_seq_len = max_seq_len;
    m_.head_num = head_num;
    m_.size_per_head = size_per_head;
    m_.inter_size = inter_size;
    m_.d_model = d_model;
    m_.num_layer = num_layer;
    m_.vocab_size = vocab_size;
    m_.sm = sm;
    m_.q_scaling = q_scaling;
    m_.useFP16 = (bool)useFP16;
    m_.batch_size = m_.max_batch_size;
    m_.seq_len = m_.max_seq_len;

    CreateFT();
}

void WenetDecoderPlugin::CreateFT()
{
    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtInit(&cusparseltHandle_));
#endif

    // Wenet EncoderWeight
    std::string paraFilePath = "/target/python/dec/bin_model/";
    if (m_.useFP16) {
        // m_.attention_type = AttentionType::UNFUSED_MHA;  // when use FP16, only this type works till v5.0-dev
        pWenetDecoderWeightHalf_ =
            new WenetDecoderWeight<half>(m_.head_num, m_.size_per_head, m_.inter_size, m_.num_layer, m_.vocab_size);
        pWenetDecoderWeightHalf_->loadModel(paraFilePath);
    }
    else {
        // m_.attention_type = getAttentionType<float>(m_.size_per_head, getSMVersion(), m_.is_remove_padding,
        // m_.max_seq_len);
        pWenetDecoderWeightFloat_ =
            new WenetDecoderWeight<float>(m_.head_num, m_.size_per_head, m_.inter_size, m_.num_layer, m_.vocab_size);
        pWenetDecoderWeightFloat_->loadModel(paraFilePath);
    }

    // Gemm file selection
    std::string gemmFileName = std::string(GEMM_CONFIG).substr(0, 11) + std::string("-SM") + std::to_string(m_.sm)
                               + std::string("-FP") + std::to_string(m_.useFP16 ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.batch_size) + std::string("-SL")
                               + std::to_string(m_.seq_len)
                               //+ std::string("-BM") + std::to_string(m_.beam_width)
                               + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
#if DEBUG_ENABLE == 1
        printf("Gemm file exist!\n");
#endif
    }
    else {
#if DEBUG_ENABLE == 1
        printf("Gemm file do not exist!\n");
#endif
        /*
                int argv[16] = {
                    0,
                    (int)m_.max_batch_size,
                    (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : (int)m_.seq_len,  // seq_len, in case of OOM
                    (int)m_.d_model,
                    (int)m_.head_num,
                    (int)m_.size_per_head,
                    (int)m_.inter_size,
                    (int)m_.d_model,
                    (int)m_.head_num,
                    (int)m_.size_per_head,
                    (int)m_.inter_size,
                    (int)m_.vocab_size,
                    (int)m_.useFP16,  // is_fp16
                    (int)m_.useFP16   // is_fp16_compute_type
                };
                wenet_gemm(argv);
                rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
        */
    }

    pCublasAlgoMap_ = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    // cublas wrapper and WenetEncoder
#ifdef SPARSITY_ENABLED
    pCublasWrapper_ = new cublasMMWrapper(
        cublasHandle_, cublasltHandle_, cusparseltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);
    m_.is_sparse = true;
#else
    pCublasWrapper_ =
        new cublasMMWrapper(cublasHandle_, cublasltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);
    m_.is_sparse = false;
#endif
    if (m_.useFP16) {
        pCublasWrapper_->setFP16GemmConfig();
        pWenetDecoderHalf_ = new WenetDecoder<half>(m_.max_batch_size,
                                                    m_.max_seq_len,
                                                    m_.head_num,
                                                    m_.size_per_head,
                                                    m_.inter_size,
                                                    m_.num_layer,
                                                    m_.vocab_size,
                                                    m_.q_scaling,
                                                    0,  // stream placeholder
                                                    pCublasWrapper_,
                                                    pAllocator_,
                                                    m_.is_free_buffer_after_forward);
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pWenetDecoderFloat_ = new WenetDecoder<float>(m_.max_batch_size,
                                                      m_.max_seq_len,
                                                      m_.head_num,
                                                      m_.size_per_head,
                                                      m_.inter_size,
                                                      m_.num_layer,
                                                      m_.vocab_size,
                                                      m_.q_scaling,
                                                      0,  // stream placeholder
                                                      pCublasWrapper_,
                                                      pAllocator_,
                                                      m_.is_free_buffer_after_forward);
    }
    PRINT_DECODER(m_.useFP16)
}

WenetDecoderPlugin::WenetDecoderPlugin(const std::string& name, const void* buffer, size_t length): name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));

    CreateFT();
}

WenetDecoderPlugin::~WenetDecoderPlugin()
{
    WHERE_AM_I();
    if (pWenetDecoderWeightHalf_ != nullptr) {
        delete pWenetDecoderWeightHalf_;
    }
    if (pWenetDecoderWeightFloat_ != nullptr) {
        delete pWenetDecoderWeightFloat_;
    }
    if (pWenetDecoderHalf_ != nullptr) {
        delete pWenetDecoderHalf_;
    }
    if (pWenetDecoderFloat_ != nullptr) {
        delete pWenetDecoderFloat_;
    }
    delete pCublasAlgoMap_;
    delete pCublasWrapperMutex_;
    delete pCublasWrapper_;
    delete pAllocator_;
}

size_t WenetDecoderPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void WenetDecoderPlugin::serialize(void* buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
}

IPluginV2DynamicExt* WenetDecoderPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new WenetDecoderPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int WenetDecoderPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType WenetDecoderPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    WHERE_AM_I();
    //return m_.useFP16 ? DataType::kHALF : DataType::kFLOAT;
    return DataType::kFLOAT;
}

bool WenetDecoderPlugin::supportsFormatCombination(int pos,
                                                   const PluginTensorDesc* inOut,
                                                   int nbInputs,
                                                   int nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res = false;

    switch (pos) {
        case 1:
        case 3:
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 0:
        case 2:
            res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 4:
            //res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
            //      && (inOut[pos].format == TensorFormat::kLINEAR);
            res = (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        default:  // should NOT be here!
            ;
    }
#if DEBUG_ENABLE == 1
    printf("Dim(");
    for (int i = 0; i < 3; i++) {
        printf("%d,", inOut[i].dims.nbDims);
    }
    printf("),");
    printf("pos=%d,res=%d,format(%d,%d,%d),type(%d,%d,%d),",
           pos,
           int(res),
           int(inOut[0].format),
           int(inOut[1].format),
           int(inOut[2].format),
           int(inOut[0].type),
           int(inOut[1].type),
           int(inOut[2].type));
    printf("kLINEAR=%d,float=%d,half=%d,int8=%d,int32=%d,bool=%d\n",
           int(TensorFormat::kLINEAR),
           int(DataType::kFLOAT),
           int(DataType::kHALF),
           int(DataType::kINT8),
           int(DataType::kINT32),
           int(DataType::kBOOL));
#endif
    return res;
}

DimsExprs WenetDecoderPlugin::getOutputDimensions(int index,
                                                  const DimsExprs* pInputDim,
                                                  int nInputDim,
                                                  IExprBuilder& exprBuilder) noexcept
{
    WHERE_AM_I();
    
    DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = pInputDim[0].d[0];
    ret.d[1] = pInputDim[0].d[1];
    ret.d[2] = exprBuilder.constant(m_.vocab_size);
    return ret;
    //return pInputDim[0];
}

void WenetDecoderPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                         int nbInput,
                                         const DynamicPluginTensorDesc* out,
                                         int nbOutput) noexcept
{
    WHERE_AM_I();
    PRINT_DECODER(int(out[0].desc.type))
}

size_t WenetDecoderPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                            int32_t nbInputs,
                                            const PluginTensorDesc* outputs,
                                            int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

void WenetDecoderPlugin::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* WenetDecoderPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* WenetDecoderPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return DECODER_NAME;
}

const char* WenetDecoderPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return DECODER_VERSION;
}

int WenetDecoderPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void WenetDecoderPlugin::terminate() noexcept
{
    WHERE_AM_I();
}

void WenetDecoderPlugin::destroy() noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    cublasDestroy(cublasHandle_);
    cublasLtDestroy(cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparseltHandle_);
#endif
    delete this;
}

void WenetDecoderPlugin::attachToContext(cudnnContext* /*cudnn*/,
                                         cublasContext* /*cublas*/,
                                         IGpuAllocator* /*allocator*/) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
}

void WenetDecoderPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
}

int WenetDecoderPlugin::enqueue(const PluginTensorDesc* inputDesc,
                                const PluginTensorDesc* outputDesc,
                                const void* const* inputs,
                                void* const* outputs,
                                void* workspace,
                                cudaStream_t stream) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    FT_CHECK(inputDesc[0].dims.nbDims == 3);
    FT_CHECK(inputDesc[2].dims.nbDims == 3);

    const size_t batch_size = inputDesc[0].dims.d[0];
    const size_t seq_len1 = inputDesc[0].dims.d[1];
    const size_t d_model = inputDesc[0].dims.d[2];

    const size_t batch_size2 = inputDesc[2].dims.d[0];
    const size_t seq_len2 = inputDesc[2].dims.d[1];

    m_.batch_size = batch_size;
    m_.seq_len = seq_len1;
    m_.d_model = d_model;

    PRINT_DECODER(outputDesc[0].type)

    cublasSetStream(cublasHandle_, stream);
    pCublasWrapper_->setStream(stream);

    if (m_.useFP16) {
        std::vector<Tensor> inputTensor{
            Tensor{MEMORY_GPU, TYPE_FP16, std::vector<size_t>{batch_size, seq_len1, d_model}, (half*)inputs[0]},
            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, (int*)inputs[1]},
            Tensor{MEMORY_GPU, TYPE_FP16, std::vector<size_t>{batch_size2, seq_len2, d_model}, (half*)inputs[2]},
            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size2}, (int*)inputs[3]}};

        std::vector<Tensor> outputTensor{
            //Tensor{MEMORY_GPU, TYPE_FP16, std::vector<size_t>{batch_size, seq_len1, d_model}, (half*)outputs[0]}};
            Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{batch_size, seq_len1, m_.vocab_size}, (float*)outputs[0]}};
        pWenetDecoderHalf_->setStream(stream);
        pWenetDecoderHalf_->forward(&outputTensor, &inputTensor, pWenetDecoderWeightHalf_);
    }
    else {
        std::vector<Tensor> inputTensor{
            Tensor{MEMORY_GPU, TYPE_FP16, std::vector<size_t>{batch_size, seq_len1, d_model}, (float*)inputs[0]},
            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, (int*)inputs[1]},
            Tensor{MEMORY_GPU, TYPE_FP16, std::vector<size_t>{batch_size2, seq_len2, d_model}, (float*)inputs[2]},
            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size2}, (int*)inputs[3]}};

        std::vector<Tensor> outputTensor{
            //Tensor{MEMORY_GPU, TYPE_FP16, std::vector<size_t>{batch_size, seq_len1, d_model}, (float*)outputs[0]}};
            Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{batch_size, seq_len1, m_.vocab_size}, (float*)outputs[0]}};

        pWenetDecoderFloat_->setStream(stream);
        pWenetDecoderFloat_->forward(&outputTensor, &inputTensor, pWenetDecoderWeightFloat_);
    }
    return 0;
}

// class WenetDecoderPluginCreator --------------------------------------------------------------------
PluginFieldCollection WenetDecoderPluginCreator::fc_{};
std::vector<PluginField> WenetDecoderPluginCreator::attr_{
    {"max_batch_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"max_seq_len", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"head_num", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"size_per_head", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"inter_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"num_layer", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"sm", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"useFP16", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"vocab_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"int8_mode", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"q_scaling", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0}};

WenetDecoderPluginCreator::WenetDecoderPluginCreator()
{
    WHERE_AM_I();
    fc_.nbFields = attr_.size();
    fc_.fields = attr_.data();
}

WenetDecoderPluginCreator::~WenetDecoderPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2* WenetDecoderPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    int max_batch_size = 128;
    int max_seq_len = 384;
    int head_num = 8;
    int size_per_head = 32;
    int d_model = head_num * size_per_head;
    int inter_size = d_model * 4;
    int num_layer = 12;
    int vocab_size = 4233;
    int sm = -1;
    float q_scaling = 1.0f / (sqrt(size_per_head) * 1.0f);
    int useFP16 = 0;
    int int8_mode = 0;

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm = prop.major * 10 + prop.minor;

    std::map<std::string, int*> name2p{
        {"max_batch_size", &max_batch_size},
        {"max_seq_len", &max_seq_len},
        {"head_num", &head_num},
        {"size_per_head", &size_per_head},
        {"inter_size", &inter_size},
        {"num_layer", &num_layer},
        {"sm", &sm},
        {"useFP16", &useFP16},
        {"vocab_size", &vocab_size},
        {"int8_mode", &int8_mode},
    };
    for (int i = 0; i < fc->nbFields; i++) {
        if (!strcmp(fc->fields[i].name, "q_scaling")) {
            q_scaling = *(float*)fc->fields[i].data;
        }
        else if (name2p.find(fc->fields[i].name) != name2p.end()) {
            *name2p[fc->fields[i].name] = *(int*)fc->fields[i].data;
        }
    }
    return new WenetDecoderPlugin(name,
                                  max_batch_size,
                                  max_seq_len,
                                  head_num,
                                  size_per_head,
                                  inter_size,
                                  head_num * size_per_head,
                                  num_layer,
                                  vocab_size,
                                  sm,
                                  q_scaling,
                                  useFP16);
}

IPluginV2*
WenetDecoderPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new WenetDecoderPlugin(name, serialData, serialLength);
}

void WenetDecoderPluginCreator::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* WenetDecoderPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* WenetDecoderPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return DECODER_NAME;
}

const char* WenetDecoderPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return DECODER_VERSION;
}

const PluginFieldCollection* WenetDecoderPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(WenetDecoderPluginCreator);

}  // namespace nvinfer1
