#include "mimo_for_causal_lm.hpp"
#include "../models_registry.hpp"
#include "infinicore/ops.hpp"
#include <stdexcept>
#include <string>

namespace infinilm::models::mimo {

MiMoMTPLayers::MiMoMTPLayers(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device,
                             size_t layer_idx)
    : hidden_size_(model_config->get<size_t>("hidden_size")) {
    const auto &dtype{model_config->get_dtype()};
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(token_layernorm, hidden_size_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(hidden_layernorm, hidden_size_, rms_norm_eps, dtype, device);
    // input_proj fuses [hidden_layernorm(hidden) | token_layernorm(embed)] -> hidden.
    INFINICORE_NN_MODULE_INIT(input_proj, hidden_size_ * 2, hidden_size_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size_, rms_norm_eps, dtype, device);
    self_attn_ = this->register_module<MiMoAttention>("self_attn", model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size_, rms_norm_eps, dtype, device);
    mlp_ = this->register_module<MiMoMLP>("mlp", model_config, device);
    INFINICORE_NN_MODULE_INIT(final_layernorm, hidden_size_, rms_norm_eps, dtype, device);
}

infinicore::Tensor MiMoMTPLayers::forward(const infinicore::Tensor &input_embeds,
                                          const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &positions) const {
    auto token_normed = token_layernorm_->forward(input_embeds);
    auto hidden_normed = hidden_layernorm_->forward(hidden_states);

    // Fuse [hidden_layernorm(hidden) | token_layernorm(embed)] along the last dim.
    auto fused_shape = token_normed->shape();
    fused_shape.back() = hidden_size_ * 2;
    auto fused_input = infinicore::Tensor::empty(fused_shape, token_normed->dtype(), token_normed->device());
    fused_input->narrow({{fused_shape.size() - 1, 0, hidden_size_}})->copy_from(hidden_normed);
    fused_input->narrow({{fused_shape.size() - 1, hidden_size_, hidden_size_}})->copy_from(token_normed);

    auto hidden = input_proj_->forward(fused_input);

    // Decoder block (attention + MLP with residuals).
    auto residual = hidden;
    hidden = input_layernorm_->forward(hidden);
    hidden = self_attn_->forward(positions, hidden);
    hidden = infinicore::op::add(residual, hidden);
    residual = hidden;
    hidden = post_attention_layernorm_->forward(hidden);
    hidden = mlp_->forward(hidden);
    hidden = infinicore::op::add(residual, hidden);

    hidden = final_layernorm_->forward(hidden);
    return hidden;
}

std::shared_ptr<infinilm::config::ModelConfig> create_mimo_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("mimo" != model_type) {
        throw std::runtime_error(
            "infinilm::models::mimo::create_mimo_model_config: model_type is not mimo");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    // MiMo's config omits head_dim; derive it from hidden_size / num_attention_heads.
    if (!config_json.contains("head_dim")) {
        size_t head_dim = model_config->get<size_t>("hidden_size")
                        / model_config->get<size_t>("num_attention_heads");
        config_json["head_dim"] = head_dim;
    }

    return model_config;
}

} // namespace infinilm::models::mimo

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    mimo,
    infinilm::models::mimo::MiMoForCausalLM,
    infinilm::models::mimo::create_mimo_model_config);
} // namespace
