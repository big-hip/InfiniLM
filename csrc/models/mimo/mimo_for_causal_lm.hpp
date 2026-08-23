#pragma once

#include "../../layers/common_modules.hpp"
#include "../../models/infinilm_model.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::mimo {

using MiMoMLP = infinilm::layers::MLP;
using MiMoAttention = infinilm::layers::attention::Attention;
using MiMoDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<MiMoAttention, MiMoMLP>;

/**
 * @brief MiMo Multi-Token Prediction (MTP) layer.
 *
 * A single transformer block that, given the previous token's hidden state and
 * the next token's embedding, fuses them through an input projection and runs
 * one attention + MLP block followed by a final layernorm. Its output is fed
 * to the language-model head to predict the next-next token (DeepSeek-V3
 * style MTP), enabling speculative decoding.
 *
 * Weight keys match `model.mtp_layers.N.*` in the MiMo checkpoint:
 * token_layernorm, hidden_layernorm, input_proj, input_layernorm, self_attn,
 * post_attention_layernorm, mlp, final_layernorm.
 */
class MiMoMTPLayers : public infinicore::nn::Module {
public:
    /**
     * @param model_config MiMo model config.
     * @param device       Device to create tensors on.
     * @param layer_idx    Index of this MTP layer's KV-cache slot. It is offset
     *                     past the backbone layers (e.g. `num_hidden_layers + i`)
     *                     so the MTP attention never reuses a backbone slot.
     */
    MiMoMTPLayers(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device,
                  size_t layer_idx);

    /**
     * @brief Run the MTP layer.
     * @param input_embeds Embedding of the next-token input ids.
     * @param hidden_states Final hidden states of the main backbone.
     * @param positions Position ids of the next-token input ids.
     */
    infinicore::Tensor forward(const infinicore::Tensor &input_embeds,
                               const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &positions) const;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, token_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, hidden_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, input_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(MiMoAttention, self_attn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiMoMLP, mlp);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, final_layernorm);
};

/**
 * @brief MiMo model: a Qwen2 backbone plus a stack of MTP layers.
 *
 * The backbone reuses the shared TextModel; the MTP layers are registered as
 * `mtp_layers` so their weights load from `model.mtp_layers.N.*`.
 */
class MiMoModel : public infinilm::layers::causal_lm_templates::TextModel<MiMoDecoderLayer> {
public:
    using Base = infinilm::layers::causal_lm_templates::TextModel<MiMoDecoderLayer>;

    MiMoModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              const infinicore::Device &device)
        : Base(model_config, device) {
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        // Default 0 so a config without an MTP head stays consistent with the
        // KV-cache allocation (InfinilmModel reserves num_nextn_predict_layers
        // extra slots); MiMo-7B ships with num_nextn_predict_layers=1.
        size_t num_mtp_layers = model_config->get_or<size_t>("num_nextn_predict_layers", 0);
        mtp_layers_.reserve(num_mtp_layers);
        // Each MTP layer owns its own KV-cache slot past the backbone layers.
        for (size_t i = 0; i < num_mtp_layers; ++i) {
            mtp_layers_.push_back(this->register_module<MiMoMTPLayers>(
                "mtp_layers." + std::to_string(i), model_config, device, num_hidden_layers + i));
        }
    }

    /**
     * @brief Run the MTP path: embed input_ids with the shared embedding, fuse
     * with the target-model hidden states, and pass through all MTP layers.
     */
    infinicore::Tensor forward_mtp(const infinicore::Tensor &input_ids,
                                   const infinicore::Tensor &positions,
                                   const infinicore::Tensor &target_hidden_states) const {
        auto input_embeds = this->embed_tokens(input_ids);
        auto hidden_states = target_hidden_states;
        for (const auto &layer : mtp_layers_) {
            hidden_states = layer->forward(input_embeds, hidden_states, positions);
        }
        return hidden_states;
    }

protected:
    INFINICORE_NN_MODULE_VEC(MiMoMTPLayers, mtp_layers);
};

/**
 * @brief MiMo Causal LM.
 *
 * Reuses the shared Qwen2 backbone forward (TextCausalLM). When
 * `Input::target_hidden_states` is supplied (e.g. by a speculative / MTP
 * caller), the MTP path is taken instead: MTP layers run on top of the target
 * hidden states and produce next-next-token logits through the shared lm_head.
 */
class MiMoForCausalLM : public infinilm::layers::causal_lm_templates::TextCausalLM<MiMoModel> {
public:
    using Base = infinilm::layers::causal_lm_templates::TextCausalLM<MiMoModel>;

    MiMoForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device)
        : Base(model_config, device) {}

    Output forward(const Input &input) const override {
        if (input.target_hidden_states.has_value()) {
            auto hidden_states = this->model_->forward_mtp(
                input.input_ids.value(),
                input.position_ids.value(),
                input.target_hidden_states.value());

            // Mirror TextCausalLM::forward: when not asked to keep every
            // position, reduce logits to the last token of each request.
            auto lm_head_input = hidden_states;
            if (!input.sample_all_positions && input.input_offsets.has_value()) {
                const size_t num_requests = input.input_offsets.value()->numel() - 1;
                const bool is_packed_prefill = hidden_states->ndim() == 3
                                            && hidden_states->size(0) == 1
                                            && hidden_states->size(1) > num_requests;
                if (is_packed_prefill) {
                    lm_head_input = infinicore::Tensor::empty(
                        {1, num_requests, hidden_states->size(2)},
                        hidden_states->dtype(),
                        hidden_states->device());
                    infinicore::op::select_last_token_hidden_(
                        lm_head_input, hidden_states, input.input_offsets.value());
                }
            }

            auto logits = this->logits_from_hidden(lm_head_input);
            return {logits, hidden_states};
        }
        return Base::forward(input);
    }
};

std::shared_ptr<infinilm::config::ModelConfig> create_mimo_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::mimo
