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

#include "EncoderPlugin.h"

using namespace fastertransformer;

namespace nvinfer1 {

// class WenetEncoderPlugin ---------------------------------------------------------------------------
WenetEncoderPlugin::WenetEncoderPlugin(const std::string& name,
                                       size_t max_batch_size,
                                       size_t max_seq_len,
                                       size_t head_num,
                                       size_t size_per_head,
                                       size_t inter_size,
                                       size_t d_model,
                                       size_t num_layer,
                                       size_t vocab_size,
                                       size_t beam_width,
                                       int sm,
                                       float q_scaling,
                                       int useFP16,
                                       int int8_mode):
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
    m_.beam_width = beam_width;
    m_.sm = sm;
    m_.q_scaling = q_scaling;
    m_.useFP16 = (bool)useFP16;
    m_.int8_mode = int8_mode;
    m_.batch_size = m_.max_batch_size;
    m_.seq_len = m_.max_seq_len;

    CreateFT();
}

void WenetEncoderPlugin::CreateFT()
{
    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtInit(&cusparseltHandle_));
#endif

    // Wenet EncoderWeight
    std::string paraFilePath = "/target/python/enc/bin_model/";
    if (m_.useFP16) {
        m_.attention_type = AttentionType::UNFUSED_MHA;  // when use FP16, only this type works till v5.0-dev
        pWenetEncoderWeightHalf_ = new WenetEncoderWeight<half>(
            m_.head_num, m_.size_per_head, m_.inter_size, m_.d_model, m_.vocab_size, m_.num_layer, m_.int8_mode);
        pWenetEncoderWeightHalf_->loadModel(paraFilePath);
    }
    else {
        m_.attention_type =
            getAttentionType<float>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len);
        pWenetEncoderWeightFloat_ = new WenetEncoderWeight<float>(
            m_.head_num, m_.size_per_head, m_.inter_size, m_.d_model, m_.vocab_size, m_.num_layer, m_.int8_mode);
        pWenetEncoderWeightFloat_->loadModel(paraFilePath);
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


    if(m_.int8_mode==0)
    {
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
    }
    else
    {
    // cublas wrapper and WenetEncoder
#ifdef SPARSITY_ENABLED
    pCublasWrapper_ = new cublasINT8MMWrapper(
        cublasltHandle_, cusparseltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, false);
    m_.is_sparse = true;
#else
    pCublasWrapper_ =
        new cublasINT8MMWrapper(cublasHandle_, cublasltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, false);
    m_.is_sparse = false;
#endif
    }
    if (m_.useFP16) {
        pCublasWrapper_->setFP16GemmConfig();

        pWenetEncoderHalf_ = new WenetEncoder<half>(m_.max_batch_size,
                                                    m_.max_seq_len,
                                                    m_.head_num,
                                                    m_.size_per_head,
                                                    m_.inter_size,
                                                    m_.d_model,
                                                    m_.num_layer,
                                                    m_.vocab_size,
                                                    m_.beam_width,
                                                    m_.sm,
                                                    m_.q_scaling,
                                                    m_.int8_mode,
                                                    0,  // stream placeholder
                                                    pCublasWrapper_,
                                                    pAllocator_,
                                                    m_.is_free_buffer_after_forward,
                                                    m_.attention_type,
                                                    m_.is_sparse,
                                                    m_.activation_type);
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pWenetEncoderFloat_ = new WenetEncoder<float>(m_.max_batch_size,
                                                      m_.max_seq_len,
                                                      m_.head_num,
                                                      m_.size_per_head,
                                                      m_.inter_size,
                                                      m_.d_model,
                                                      m_.num_layer,
                                                      m_.vocab_size,
                                                      m_.beam_width,
                                                      m_.sm,
                                                      m_.q_scaling,
                                                      m_.int8_mode,
                                                      0,  // stream placeholder
                                                      pCublasWrapper_,
                                                      pAllocator_,
                                                      m_.is_free_buffer_after_forward,
                                                      m_.attention_type,
                                                      m_.is_sparse,
                                                      m_.activation_type);
    }
    PRINT_ENCODER(m_.useFP16)
}

WenetEncoderPlugin::WenetEncoderPlugin(const std::string& name, const void* buffer, size_t length): name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));

    CreateFT();
}

WenetEncoderPlugin::~WenetEncoderPlugin()
{
    WHERE_AM_I();
    if (pWenetEncoderWeightHalf_ != nullptr) {
        delete pWenetEncoderWeightHalf_;
    }
    if (pWenetEncoderWeightFloat_ != nullptr) {
        delete pWenetEncoderWeightFloat_;
    }
    if (pWenetEncoderHalf_ != nullptr) {
        delete pWenetEncoderHalf_;
    }
    if (pWenetEncoderFloat_ != nullptr) {
        delete pWenetEncoderFloat_;
    }
    delete pCublasAlgoMap_;
    delete pCublasWrapperMutex_;
    delete pCublasWrapper_;
    delete pAllocator_;
}

size_t WenetEncoderPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void WenetEncoderPlugin::serialize(void* buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
}

IPluginV2DynamicExt* WenetEncoderPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new WenetEncoderPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int WenetEncoderPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 5;
}

DataType WenetEncoderPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    WHERE_AM_I();
    if (index == 1)
        return DataType::kINT32;
    else if (index == 2)
        return DataType::kFLOAT;
    else
        return m_.useFP16 ? DataType::kHALF : DataType::kFLOAT;
}

bool WenetEncoderPlugin::supportsFormatCombination(int pos,
                                                   const PluginTensorDesc* inOut,
                                                   int nbInputs,
                                                   int nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res = false;

    switch (pos) {
        case 0:  // encoder_in
            res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 1:  // speech_length
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 2:  // pos_emb
            res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 3:  // speech
            res = (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 4:  // encoder_out
            res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT))
                  && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 5:  // encoder_out_length
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 6:  // ctc_log_probs
            res = (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 7:  // beam_log_probs
            res = (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 8:  // beam_log_probs_idx
            res = (inOut[pos].type == DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
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

DimsExprs WenetEncoderPlugin::getOutputDimensions(int index,
                                                  const DimsExprs* pInputDim,
                                                  int nInputDim,
                                                  IExprBuilder& exprBuilder) noexcept
{
    WHERE_AM_I();
    /*
    DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = pInputDim[0].d[0];
    ret.d[1] = pInputDim[0].d[1];
    ret.d[2] = exprBuilder.constant(m_.d_model);
    */
    if (index < 2)
        return pInputDim[index];
    else if (index == 2) {
        DimsExprs ret;
        ret.nbDims = 3;
        ret.d[0] = pInputDim[0].d[0];
        ret.d[1] = pInputDim[0].d[1];
        ret.d[2] = exprBuilder.constant(m_.vocab_size);
        return ret;
    }
    else {
        DimsExprs ret;
        ret.nbDims = 3;
        ret.d[0] = pInputDim[0].d[0];
        ret.d[1] = pInputDim[0].d[1];
        ret.d[2] = exprBuilder.constant(m_.beam_width);
        return ret;
    }
}

void WenetEncoderPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                         int nbInput,
                                         const DynamicPluginTensorDesc* out,
                                         int nbOutput) noexcept
{
    WHERE_AM_I();
    PRINT_ENCODER(int(out[0].desc.type))
}

size_t WenetEncoderPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                            int32_t nbInputs,
                                            const PluginTensorDesc* outputs,
                                            int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

void WenetEncoderPlugin::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* WenetEncoderPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* WenetEncoderPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return ENCODER_NAME;
}

const char* WenetEncoderPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return ENCODER_VERSION;
}

int WenetEncoderPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void WenetEncoderPlugin::terminate() noexcept
{
    WHERE_AM_I();
}

void WenetEncoderPlugin::destroy() noexcept
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

void WenetEncoderPlugin::attachToContext(cudnnContext* /*cudnn*/,
                                         cublasContext* /*cublas*/,
                                         IGpuAllocator* /*allocator*/) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
}

void WenetEncoderPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
}

int WenetEncoderPlugin::enqueue(const PluginTensorDesc* inputDesc,
                                const PluginTensorDesc* outputDesc,
                                const void* const* inputs,
                                void* const* outputs,
                                void* workspace,
                                cudaStream_t stream) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.batch_size = inputDesc[0].dims.d[0];
    m_.seq_len = inputDesc[0].dims.d[1];
    PRINT_ENCODER(outputDesc[0].type)
    // return 0;
    //  std::cout << "emb_pos shape:" << inputDesc[2].dims.d[0] << "," << inputDesc[2].dims.d[1] << "," <<
    //  inputDesc[2].dims.d[2] << std::endl;
    // std::cout << "speech shape:" << inputDesc[3].dims.d[0] << "," << inputDesc[3].dims.d[1] << ","
    //          << inputDesc[3].dims.d[2] << std::endl;

    cublasSetStream(cublasHandle_, stream);
    pCublasWrapper_->setStream(stream);

    if (m_.useFP16) {
        std::unordered_map<std::string, Tensor> inputTensor{
            {"input_hidden_state",
             Tensor{
                 MEMORY_GPU, TYPE_FP16, std::vector<size_t>{m_.batch_size, m_.seq_len, m_.d_model}, (half*)inputs[0]}},
            {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)inputs[1]}},
            {"pos_emb",
             Tensor{
                 MEMORY_GPU, TYPE_FP16, std::vector<size_t>{m_.batch_size, m_.seq_len, m_.d_model}, (half*)inputs[2]}},
            {"speech",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, (size_t)inputDesc[3].dims.d[1], (size_t)inputDesc[3].dims.d[2]},
                    (float*)inputs[3]}}};

        std::unordered_map<std::string, Tensor> outputTensor{
            {"output_hidden_state",
             Tensor{
                 MEMORY_GPU, TYPE_FP16, std::vector<size_t>{m_.batch_size, m_.seq_len, m_.d_model}, (half*)outputs[0]}},
            {"encoder_out_lens", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)outputs[1]}},
            {"ctc_log_probs",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, m_.vocab_size},
                    (float*)outputs[2]}},
            {"beam_log_probs",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, m_.beam_width},
                    (float*)outputs[3]}},
            {"beam_log_probs_idx",
             Tensor{MEMORY_GPU,
                    TYPE_INT32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, m_.beam_width},
                    (int*)outputs[4]}},
        };
        pWenetEncoderHalf_->setStream(stream);
        pWenetEncoderHalf_->forward(&outputTensor, &inputTensor, pWenetEncoderWeightHalf_);
    }
    else {
        std::unordered_map<std::string, Tensor> inputTensor{
            {"input_hidden_state",
             Tensor{
                 MEMORY_GPU, TYPE_FP32, std::vector<size_t>{m_.batch_size, m_.seq_len, m_.d_model}, (float*)inputs[0]}},
            {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)inputs[1]}},
            {"pos_emb",
             Tensor{
                 MEMORY_GPU, TYPE_FP32, std::vector<size_t>{m_.batch_size, m_.seq_len, m_.d_model}, (float*)inputs[2]}},
            {"speech",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, (size_t)inputDesc[3].dims.d[1], (size_t)inputDesc[3].dims.d[2]},
                    (float*)inputs[3]}}};

        std::unordered_map<std::string, Tensor> outputTensor{
            {"output_hidden_state",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, m_.d_model},
                    (float*)outputs[0]}},
            {"encoder_out_lens", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)outputs[1]}},
            {"ctc_log_probs",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, m_.vocab_size},
                    (float*)outputs[2]}},
            {"beam_log_probs",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, m_.beam_width},
                    (float*)outputs[3]}},
            {"beam_log_probs_idx",
             Tensor{MEMORY_GPU,
                    TYPE_INT32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, m_.beam_width},
                    (int*)outputs[4]}},
        };

        pWenetEncoderFloat_->setStream(stream);
        pWenetEncoderFloat_->forward(&outputTensor, &inputTensor, pWenetEncoderWeightFloat_);
    }
    return 0;
}

// class WenetEncoderPluginCreator --------------------------------------------------------------------
PluginFieldCollection WenetEncoderPluginCreator::fc_{};
std::vector<PluginField> WenetEncoderPluginCreator::attr_{
    {"max_batch_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"max_seq_len", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"head_num", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"size_per_head", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"inter_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"d_model", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"num_layer", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"sm", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"useFP16", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"vocab_size", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"int8_mode", nullptr, nvinfer1::PluginFieldType::kINT32, 0},
    {"q_scaling", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0}};

WenetEncoderPluginCreator::WenetEncoderPluginCreator()
{
    WHERE_AM_I();
    fc_.nbFields = attr_.size();
    fc_.fields = attr_.data();
}

WenetEncoderPluginCreator::~WenetEncoderPluginCreator()
{
    WHERE_AM_I();
}

IPluginV2* WenetEncoderPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
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
    int beam_width = 10;
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
        {"d_model", &d_model},
        {"num_layer", &num_layer},
        {"sm", &sm},
        {"useFP16", &useFP16},
        {"vocab_size", &vocab_size},
        {"int8_mode", &int8_mode}
    };
    for (int i = 0; i < fc->nbFields; i++) {
        if (!strcmp(fc->fields[i].name, "q_scaling")) {
            q_scaling = *(float*)fc->fields[i].data;
        }
        else if (name2p.find(fc->fields[i].name) != name2p.end()) {
            *name2p[fc->fields[i].name] = *(int*)fc->fields[i].data;
        }
    }
    //std::cout << "########################################" << vocab_size<<std::endl;

    return new WenetEncoderPlugin(name,
                                  max_batch_size,
                                  max_seq_len,
                                  head_num,
                                  size_per_head,
                                  inter_size,
                                  d_model,
                                  num_layer,
                                  vocab_size,
                                  beam_width,
                                  sm,
                                  q_scaling,
                                  useFP16,
                                  int8_mode);
}

IPluginV2*
WenetEncoderPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new WenetEncoderPlugin(name, serialData, serialLength);
}

void WenetEncoderPluginCreator::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* WenetEncoderPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* WenetEncoderPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return ENCODER_NAME;
}

const char* WenetEncoderPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return ENCODER_VERSION;
}

const PluginFieldCollection* WenetEncoderPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(WenetEncoderPluginCreator);

}  // namespace nvinfer1
