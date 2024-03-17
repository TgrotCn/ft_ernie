/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/Ernie/Ernie.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"

namespace fastertransformer {

template<typename T>
void Ernie<T>::initialize()
{
    if (std::is_same<T, half>::value
        && (attention_type_ == AttentionType::FUSED_MHA || attention_type_ == AttentionType::FUSED_PADDED_MHA)) {
        fused_attention_layer_ = new FusedAttentionLayer<T>(0,
                                                            0,
                                                            head_num_ / tensor_para_.world_size_,
                                                            size_per_head_,
                                                            head_num_ * size_per_head_,
                                                            sm_,
                                                            q_scaling_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            sparse_);
    }
    unfused_attention_layer_ = new UnfusedAttentionLayer<T>(0,
                                                            0,
                                                            head_num_ / tensor_para_.world_size_,
                                                            size_per_head_,
                                                            head_num_ * size_per_head_,
                                                            q_scaling_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            sparse_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU;
    if (activation_type_ == ActivationType::Gelu) {
        ffn_layer_ = new TensorParallelGeluFfnLayer<T>(0,
                                                       0,
                                                       head_num_,
                                                       size_per_head_,
                                                       0,  // expert_num
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       0,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Relu) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(0,
                                                       0,
                                                       head_num_,
                                                       size_per_head_,
                                                       0,  // expert_num
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       0,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
}

template<typename T>
Ernie<T>::Ernie(size_t                              max_batch_size,
              size_t                              max_seq_len,
              size_t                              head_num,
              size_t                              size_per_head,
              size_t                              inter_size, // ffn层 yincangsize
              size_t                              num_layer,
              int                                 sm,
              float                               q_scaling,
              cudaStream_t                        stream,
              cublasMMWrapper*                    cublas_wrapper,
              IAllocator*                         allocator,
              bool                                is_free_buffer_after_forward,
              AttentionType                       attention_type,
              bool                                sparse,
              ActivationType                      activation_type,
              LayerNormType                       layernorm_type,
              NcclParam                           tensor_para,
              NcclParam                           pipeline_para,
              std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
              bool                                enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    sm_(sm),
    q_scaling_(q_scaling),
    attention_type_(attention_type),
    sparse_(sparse),
    activation_type_(activation_type),
    layernorm_type_(layernorm_type),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
Ernie<T>::Ernie(size_t           max_batch_size,
              size_t           max_seq_len,
              size_t           head_num,
              size_t           size_per_head,
              size_t           inter_size,
              size_t           num_layer,
              int              sm,
              float            q_scaling,
              cudaStream_t     stream,
              cublasMMWrapper* cublas_wrapper,
              IAllocator*      allocator,
              bool             is_free_buffer_after_forward,
              AttentionType    attention_type,
              bool             sparse,
              ActivationType   activation_type,
              LayerNormType    layernorm_type):
    Ernie(max_batch_size,
         max_seq_len,
         head_num,
         size_per_head,
         inter_size,
         num_layer,
         sm,
         q_scaling,
         stream,
         cublas_wrapper,
         allocator,
         is_free_buffer_after_forward,
         attention_type,
         sparse,
         activation_type,
         layernorm_type,
         NcclParam(0, 1),
         NcclParam(0, 1),
         nullptr,
         false)
{
}

template<typename T>
Ernie<T>::Ernie(Ernie<T> const& ernie):
    Ernie(0,
         0,
         ernie.head_num_,
         ernie.size_per_head_,
         ernie.inter_size_,
         ernie.num_layer_,
         ernie.sm_,
         ernie.q_scaling_,
         ernie.stream_,
         ernie.cublas_wrapper_,
         ernie.allocator_,
         ernie.is_free_buffer_after_forward_,
         ernie.attention_type_,
         ernie.sparse_,
         ernie.activation_type_,
         ernie.layernorm_type_,
         ernie.tensor_para_,
         ernie.pipeline_para_,
         ernie.custom_all_reduce_comm_,
         ernie.enable_custom_all_reduce_) { }

template<typename T>
Ernie<T>::~Ernie()
{
    if (fused_attention_layer_ != nullptr) {
        delete fused_attention_layer_;
    }
    delete unfused_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void Ernie<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void Ernie<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);
    padding_offset_         = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false);
    trt_mha_padding_offset_ =
        (int*)allocator_->reMalloc(trt_mha_padding_offset_, sizeof(int) * (2 * batch_size + 1), false);

    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false);

    ernie_in_buffer_ =
        (T*)allocator_->reMalloc(ernie_in_buffer_, sizeof(T) * batch_size * seq_len * head_num_ * size_per_head_, false);
    attn_out_buf_    = (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    ernie_out_buffer_ = (T*)allocator_->reMalloc(
        ernie_out_buffer_, sizeof(T) * batch_size * seq_len * head_num_ * size_per_head_, false);

    if (layernorm_type_ == LayerNormType::post_layernorm) {
        normed_from_tensor_  = nullptr;
        normed_attn_out_buf_ = nullptr;
    }
    else {
        normed_from_tensor_ =
            (T*)allocator_->reMalloc(normed_from_tensor_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
        normed_attn_out_buf_ =
            (T*)allocator_->reMalloc(normed_attn_out_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void Ernie<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&h_pinned_token_num_ptr_), true);
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&trt_mha_padding_offset_));

        allocator_->free((void**)(&attention_mask_));
        allocator_->free((void**)(&ernie_in_buffer_));
        allocator_->free((void**)(&attn_out_buf_));
        allocator_->free((void**)(&ernie_out_buffer_));

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            normed_from_tensor_  = nullptr;
            normed_attn_out_buf_ = nullptr;
        }
        else {
            allocator_->free((void**)(&normed_from_tensor_));
            allocator_->free((void**)(&normed_attn_out_buf_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool Ernie<T>::isValidLayerParallelId(uint l)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool Ernie<T>::isFirstLayerParallelId(uint l)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool Ernie<T>::isLastLayerParallelId(uint l)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int Ernie<T>::getFirstLayerParallelId()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
void Ernie<T>::forward(std::vector<Tensor>*       output_tensors,
                       const std::vector<Tensor>* input_tensors,
                       const ErnieWeight<T>*       ernie_weights)
{
    TensorMap input_tensors_map =
        TensorMap({{"input_hidden_state", input_tensors->at(0)}, {"sequence_lengths", input_tensors->at(1)}});
    TensorMap output_tensors_map = TensorMap({{"output_hidden_state", output_tensors->at(0)}});
    forward(&output_tensors_map, &input_tensors_map, ernie_weights);
}

template<typename T>
void Ernie<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors, const ErnieWeight<T>* ernie_weights)
{
    // input_tensors:
    //      input_hidden_state [batch, seqlen, hidden]
    //      sequence_lengths [batch]
    // output tensors:
    //      output_hidden_state [batch, seqlen, hidden]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    const size_t request_batch_size = input_tensors->at("input_hidden_state").shape[0];
    const size_t request_seq_len    = input_tensors->at("input_hidden_state").shape[1];
    FT_CHECK(input_tensors->size() >= 2);
    FT_CHECK(request_batch_size == input_tensors->at("sequence_lengths").shape[0]);
    FT_CHECK(input_tensors->at("input_hidden_state").shape.size() == 3);
    FT_CHECK(input_tensors->at("sequence_lengths").shape.size() == 1);
    allocateBuffer(request_batch_size, request_seq_len);

    const int* sequence_lengths = input_tensors->at("sequence_lengths").getPtr<int>();

    DataType     data_type        = getTensorType<T>();
    const size_t local_batch_size = getLocalBatchSize(request_batch_size, request_seq_len, pipeline_para_.world_size_);
    FT_CHECK(request_batch_size % local_batch_size == 0);
    const size_t  iteration_num  = request_batch_size / local_batch_size;
    AttentionType attention_type = attention_type_;
    if (fused_attention_layer_ == nullptr || fused_attention_layer_->isValidSeqLen(request_seq_len) == false) {
        if (attention_type == AttentionType::FUSED_MHA) {
            FT_LOG_WARNING("Because the input is invalid for fused mha, switch to unfused mha.");
            attention_type = AttentionType::UNFUSED_MHA;
        }
        else if (attention_type == AttentionType::FUSED_PADDED_MHA) {
            FT_LOG_WARNING("Because the input is invalid for fused mha, switch to unfused mha.");
            attention_type = AttentionType::UNFUSED_PADDED_MHA;
        }
    }

    for (uint ite = 0; ite < iteration_num; ite++) {
        Tensor*      padding_offset_tensor_ptr = nullptr;
        const size_t hidden_offset             = ite * local_batch_size * request_seq_len * hidden_units_;
        size_t       h_token_num               = 0;

        T* erine_input_ptr;
        T* erine_output_ptr;

        switch (attention_type) {
            case AttentionType::UNFUSED_MHA: {
                invokeBuildEncoderAttentionMask(
                    attention_mask_,
                    input_tensors->at("sequence_lengths").getPtrWithOffset<int>(ite * local_batch_size),
                    local_batch_size,
                    request_seq_len,
                    stream_);
                sync_check_cuda_error();
                invokeGetPaddingOffset(
                    h_pinned_token_num_ptr_,
                    &h_token_num,
                    padding_offset_,
                    input_tensors->at("sequence_lengths").getPtrWithOffset<int>(ite * local_batch_size),
                    local_batch_size,
                    request_seq_len,
                    stream_);

                invokeRemovePadding(ernie_in_buffer_,
                                    input_tensors->at("input_hidden_state").getPtrWithOffset<T>(hidden_offset),
                                    padding_offset_,
                                    h_token_num,
                                    head_num_ * size_per_head_,
                                    stream_);
                sync_check_cuda_error();

                padding_offset_tensor_ptr =
                    new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{h_token_num}, padding_offset_);
                ernie_input_ptr  = ernie_in_buffer_;
                ernie_output_ptr = ernie_out_buffer_;
                sync_check_cuda_error();
                break;
            }
            case AttentionType::UNFUSED_PADDED_MHA: {
                invokeBuildEncoderAttentionMask(
                    attention_mask_,
                    input_tensors->at("sequence_lengths").getPtrWithOffset<int>(ite * local_batch_size),
                    local_batch_size,
                    request_seq_len,
                    stream_);
                sync_check_cuda_error();
                h_token_num     = local_batch_size * request_seq_len;
                ernie_input_ptr  = input_tensors->at("input_hidden_state").getPtrWithOffset<T>(hidden_offset);
                ernie_output_ptr = output_tensors->at("output_hidden_state").getPtrWithOffset<T>(hidden_offset);
                padding_offset_tensor_ptr = new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{0}, nullptr);
                sync_check_cuda_error();
                break;
            }
            case AttentionType::FUSED_MHA: {
                invokeGetPaddingOffset(
                    h_pinned_token_num_ptr_,
                    &h_token_num,
                    padding_offset_,
                    input_tensors->at("sequence_lengths").getPtrWithOffset<int>(ite * local_batch_size),
                    local_batch_size,
                    request_seq_len,
                    stream_);

                invokeRemovePadding(ernie_in_buffer_,
                                    input_tensors->at("input_hidden_state").getPtrWithOffset<T>(hidden_offset),
                                    padding_offset_,
                                    h_token_num,
                                    head_num_ * size_per_head_,
                                    stream_);
                sync_check_cuda_error();

                invokeGetTrtPaddingOffset(
                    trt_mha_padding_offset_,
                    input_tensors->at("sequence_lengths").getPtrWithOffset<int>(ite * local_batch_size),
                    local_batch_size,
                    stream_);

                padding_offset_tensor_ptr = new Tensor(
                    MEMORY_GPU, TYPE_INT32, std::vector<size_t>{local_batch_size + 1}, trt_mha_padding_offset_);
                ernie_input_ptr  = ernie_in_buffer_;
                ernie_output_ptr = ernie_out_buffer_;
                sync_check_cuda_error();
                break;
            }
            case AttentionType::FUSED_PADDED_MHA: {
                h_token_num = local_batch_size * request_seq_len;
                invokeGetTrtPaddingOffset(
                    trt_mha_padding_offset_,
                    input_tensors->at("sequence_lengths").getPtrWithOffset<int>(ite * local_batch_size),
                    local_batch_size,
                    request_seq_len,
                    stream_);
                sync_check_cuda_error();
                padding_offset_tensor_ptr = new Tensor(
                    MEMORY_GPU, TYPE_INT32, std::vector<size_t>{local_batch_size * 2 + 1}, trt_mha_padding_offset_);
                ernie_input_ptr  = input_tensors->at("input_hidden_state").getPtrWithOffset<T>(hidden_offset);
                ernie_output_ptr = output_tensors->at("output_hidden_state").getPtrWithOffset<T>(hidden_offset);
                break;
            }
            default: {
                throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
            }
        }

        for (uint l = 0; l < num_layer_; l++) {
            if (isValidLayerParallelId(l) == false) {
                continue;
            }
            T* from_tensor = l == 0 ? ernie_input_ptr : ernie_output_ptr;
            T* out_tensor  = ernie_output_ptr;

            if (isFirstLayerParallelId(l) && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
                ftNcclRecv(from_tensor + h_token_num * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                           h_token_num * hidden_units_ / tensor_para_.world_size_,
                           pipeline_para_.rank_ - 1,
                           pipeline_para_,
                           stream_);
                if (tensor_para_.world_size_ > 1) {
                    ftNcclAllGather(from_tensor,
                                    from_tensor,
                                    h_token_num * hidden_units_ / tensor_para_.world_size_,
                                    tensor_para_.rank_,
                                    tensor_para_,
                                    stream_);
                }
            }
            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralLayerNorm(normed_from_tensor_,
                                       from_tensor,
                                       ernie_weights->ernie_layer_weights[l].attn_layernorm_weights.gamma,
                                       ernie_weights->ernie_layer_weights[l].attn_layernorm_weights.beta,
                                       layernorm_eps_,
                                       h_token_num,
                                       hidden_units_,
                                       (float*)nullptr,
                                       0,
                                       stream_);
                sync_check_cuda_error();
            }
            // Attention
            {
                TensorMap attn_input_tensors{
                    {"input_query",
                     Tensor{MEMORY_GPU,
                            data_type,
                            std::vector<size_t>{h_token_num, hidden_units_},
                            layernorm_type_ == LayerNormType::pre_layernorm ? normed_from_tensor_ : from_tensor}},
                    {"attention_mask",
                     Tensor{MEMORY_GPU,
                            data_type,
                            std::vector<size_t>{local_batch_size, 1, request_seq_len, request_seq_len},
                            attention_mask_}}};
                attn_input_tensors.insertIfValid("padding_offset", *padding_offset_tensor_ptr);
                TensorMap attn_output_tensors(
                    {{"hidden_features",
                      Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, attn_out_buf_}}});

                bool use_custom_all_reduce_kernel = false;
                std::vector<Tensor> hidden_features{attn_output_tensors.at("hidden_features")};
                if (enable_custom_all_reduce_ && custom_all_reduce_comm_ != nullptr) {
                    use_custom_all_reduce_kernel =
                        custom_all_reduce_comm_->swapInternalBuffer(&hidden_features, h_token_num * hidden_units_);
                    attn_output_tensors.at("hidden_features").data = hidden_features[0].data;
                }

                if (attention_type == AttentionType::FUSED_MHA || attention_type == AttentionType::FUSED_PADDED_MHA) {
                    fused_attention_layer_->forward(&attn_output_tensors,
                                                    &attn_input_tensors,
                                                    &ernie_weights->ernie_layer_weights[l].attention_weights);
                }
                else if (attention_type == AttentionType::UNFUSED_MHA
                         || attention_type == AttentionType::UNFUSED_PADDED_MHA) {
                    unfused_attention_layer_->forward(&attn_output_tensors,
                                                      &attn_input_tensors,
                                                      &ernie_weights->ernie_layer_weights[l].attention_weights);
                }

                if (tensor_para_.world_size_ > 1) {
                    if (!use_custom_all_reduce_kernel) {
                        ftNcclAllReduceSum(
                            attn_out_buf_, attn_out_buf_, h_token_num * hidden_units_, tensor_para_, stream_);
                    }
                    else {
                        custom_all_reduce_comm_->customAllReduce(h_token_num * hidden_units_, stream_);
                        attn_output_tensors.at("hidden_features").data = hidden_features[0].data;
                    }
                    sync_check_cuda_error();
                }
            }

            if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeAddBiasResidualLayerNorm(
                    attn_out_buf_,
                    from_tensor,
                    ernie_weights->ernie_layer_weights[l].attention_weights.attention_output_weight.bias,
                    ernie_weights->ernie_layer_weights[l].attn_layernorm_weights.gamma,
                    ernie_weights->ernie_layer_weights[l].attn_layernorm_weights.beta,
                    layernorm_eps_,
                    h_token_num,
                    hidden_units_,
                    stream_);
            }
            else if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralAddBiasResidualPreLayerNorm(
                    attn_out_buf_,
                    normed_attn_out_buf_,
                    attn_out_buf_,
                    from_tensor,
                    ernie_weights->ernie_layer_weights[l].ffn_layernorm_weights.gamma,
                    ernie_weights->ernie_layer_weights[l].ffn_layernorm_weights.beta,
                    ernie_weights->ernie_layer_weights[l].attention_weights.attention_output_weight.bias,
                    layernorm_eps_,
                    h_token_num,
                    hidden_units_,
                    (float*)nullptr,
                    (float*)nullptr,
                    (float*)nullptr,
                    (float*)nullptr,
                    0,
                    stream_);
            }
            sync_check_cuda_error();

            // FFN
            {
                TensorMap ffn_input_tensors(
                    {{"ffn_input",
                      Tensor{MEMORY_GPU,
                             data_type,
                             std::vector<size_t>{h_token_num, hidden_units_},
                             layernorm_type_ == LayerNormType::pre_layernorm ? normed_attn_out_buf_ : attn_out_buf_}}});
                TensorMap ffn_output_tensors(
                    {{"ffn_output",
                      Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, out_tensor}}});
                ffn_layer_->forward(
                    &ffn_output_tensors, &ffn_input_tensors, &ernie_weights->ernie_layer_weights[l].ffn_weights);
            }

            if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeAddBiasResidualLayerNorm(out_tensor,
                                               attn_out_buf_,
                                               ernie_weights->ernie_layer_weights[l].ffn_weights.output_weight.bias,
                                               ernie_weights->ernie_layer_weights[l].ffn_layernorm_weights.gamma,
                                               ernie_weights->ernie_layer_weights[l].ffn_layernorm_weights.beta,
                                               layernorm_eps_,
                                               h_token_num,
                                               hidden_units_,
                                               stream_);
            }
            else if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeAddBiasResidual(out_tensor,
                                      attn_out_buf_,
                                      ernie_weights->ernie_layer_weights[l].ffn_weights.output_weight.bias,
                                      h_token_num,
                                      hidden_units_,
                                      stream_);
            }
            sync_check_cuda_error();

            if (isLastLayerParallelId(l) && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
                && pipeline_para_.world_size_ > 1) {

                ftNcclSend(out_tensor + h_token_num * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                           h_token_num * hidden_units_ / tensor_para_.world_size_,
                           pipeline_para_.rank_ + 1,
                           pipeline_para_,
                           stream_);
            }
        }  // transformer layers

        if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralLayerNorm(ernie_output_ptr,
                                       ernie_output_ptr,
                                       ernie_weights->post_transformer_layernorm_weights.gamma,
                                       ernie_weights->post_transformer_layernorm_weights.beta,
                                       layernorm_eps_,
                                       h_token_num,
                                       hidden_units_,
                                       (float*)nullptr,
                                       0,
                                       stream_);
                sync_check_cuda_error();
            }

            // post process (rebuild padding)
            switch (attention_type) {
                case AttentionType::UNFUSED_MHA: {
                    invokeRebuildPadding(output_tensors->at("output_hidden_state").getPtrWithOffset<T>(hidden_offset),
                                         ernie_out_buffer_,
                                         padding_offset_,
                                         h_token_num,
                                         head_num_ * size_per_head_,
                                         stream_);
                    sync_check_cuda_error();
                    break;
                }
                case AttentionType::UNFUSED_PADDED_MHA: {
                    break;
                }
                case AttentionType::FUSED_MHA: {
                    invokeRebuildPadding(output_tensors->at("output_hidden_state").getPtrWithOffset<T>(hidden_offset),
                                         ernie_out_buffer_,
                                         padding_offset_,
                                         h_token_num,
                                         head_num_ * size_per_head_,
                                         stream_);
                    sync_check_cuda_error();
                    break;
                }
                case AttentionType::FUSED_PADDED_MHA: {
                    break;
                }
                default: {
                    throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
                }
            }
        }

        if (padding_offset_tensor_ptr != nullptr) {
            delete padding_offset_tensor_ptr;
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();

    if (pipeline_para_.world_size_ > 1) {
        ftNcclGroupStart();
        const int data_size = request_batch_size * request_seq_len * hidden_units_ / tensor_para_.world_size_;
        ftNcclBroadCast(output_tensors->at("output_hidden_state").getPtr<T>() + data_size * tensor_para_.rank_,
                        data_size,
                        pipeline_para_.world_size_ - 1,
                        pipeline_para_,
                        stream_);
        ftNcclGroupEnd();

        sync_check_cuda_error();
        if (tensor_para_.world_size_ > 1) {
            ftNcclAllGather(output_tensors->at("output_hidden_state").getPtr<T>(),
                            output_tensors->at("output_hidden_state").getPtr<T>(),
                            data_size,
                            tensor_para_.rank_,
                            tensor_para_,
                            stream_);
        }
        // throw errors when detected
        ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
    }
    cudaStreamSynchronize(stream_);
}

template class Ernie<float>;
template class Ernie<half>;
#ifdef ENABLE_BF16
template class Ernie<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
