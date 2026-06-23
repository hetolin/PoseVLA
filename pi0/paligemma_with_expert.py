from typing import List, Optional, Union

import torch
import torch.version
from torch import nn
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_utils import no_init_weights
from transformers.models.auto import CONFIG_MAPPING

from pi0.patch_embed import PatchEmbed_Mlp
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            # self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.bfloat16)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
            self.weight = None
        else:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None  # return in original dtype with None gate

        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        # self.dense.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly: [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)

        # Apply adaptive normalization: use model weight dtype to ensure compatibility
        # model_dtype = self.dense.weight.dtype  # Use the model's dtype (bfloat16)
        # scale = scale.to(model_dtype)
        # shift = shift.to(model_dtype)
        # gate = gate.to(model_dtype)
        # normed_inputs = normed_inputs.to(model_dtype)  # Convert normed_inputs to model dtype

        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self):
        if self.weight is not None:
            repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"
        if self.dense is not None:
            repr_str = f"eps={self.eps}, adaptive=True, cond_dim={self.cond_dim}"
        return repr_str

def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        is_knowledge_insulation: bool = False,
        pi05: bool = False,
        use_adarms: list[bool] | None = None,
        add_prior: bool = False,
        skip_init_weights: bool = False,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.is_knowledge_insulation = is_knowledge_insulation
        self.pi05 = pi05
        self.add_prior = add_prior
        self.skip_init_weights = skip_init_weights

        # use_adarms is a [paligemma, gemma_expert] pair of booleans.
        if use_adarms is None:
            use_adarms = [False, False]

        # ---- PaliGemma config ----
        if paligemma_config is None:
            # Pi0 default config for PaliGemma.
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
            self.paligemma_config.text_config.use_adarms = use_adarms[0]
            self.paligemma_config.text_config.adarms_cond_dim = (
                self.paligemma_config.text_config.hidden_size if use_adarms[0] else None
            )
        elif isinstance(paligemma_config, dict):
            # Override the default Pi0 config for PaliGemma with user-supplied values.
            paligemma_config.setdefault("model_type", "paligemma")
            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)
        else:
            self.paligemma_config = paligemma_config

        # ---- Gemma expert config ----
        if gemma_expert_config is None:
            # Pi0 default config for Gemma expert.
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
            self.gemma_expert_config.use_adarms = use_adarms[1]
            self.gemma_expert_config.adarms_cond_dim = 1024 if use_adarms[1] else None
        elif isinstance(gemma_expert_config, dict):
            # Override the default Pi0 config for the Gemma expert with user-supplied values.
            gemma_expert_config.setdefault("model_type", "gemma")
            cfg_cls = CONFIG_MAPPING[gemma_expert_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)
        else:
            self.gemma_expert_config = gemma_expert_config

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). "
                "Expected 'eager', 'fa2' or 'flex'."
            )


class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        # Skip random weight init when the full weights will be loaded later
        # via `from_pretrained` (e.g. inference / fine-tuning from a checkpoint).
        # This noticeably reduces build time and peak memory.
        if getattr(config, "skip_init_weights", False):
            with no_init_weights():
                print("[skip_init_weights] Skipping random init for paligemma / gemma_expert")
                self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
                self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
            # When tie_word_embeddings=True, retie so that lm_head.weight stays
            # shared with embed_tokens.weight after the no-init context.
            self.paligemma.tie_weights()
            self.gemma_expert.tie_weights()
        else:
            self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
            self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # Remove unused embed_tokens on the expert side (it shares PaliGemma's).
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

        self.pi05 = config.pi05

        if self.pi05:
            paligemma_cond_dim = getattr(config.paligemma_config, 'adarms_cond_dim', None) if getattr(config.paligemma_config, 'use_adarms', False) else None
            gemma_expert_cond_dim = getattr(config.gemma_expert_config, 'adarms_cond_dim', None) if getattr(config.gemma_expert_config, 'use_adarms', False) else None
            # paligemma
            self.paligemma.language_model.model.norm = GemmaRMSNorm(
                dim=self.config.paligemma_config.text_config.hidden_size,
                cond_dim=paligemma_cond_dim
            )
            # gemma_expert
            self.gemma_expert.model.norm = GemmaRMSNorm(
                dim=self.config.gemma_expert_config.hidden_size,
                cond_dim=gemma_expert_cond_dim
            )
            for layer_idx in range(self.paligemma.config.text_config.num_hidden_layers):
                # paligemma
                self.paligemma.language_model.model.layers[layer_idx].input_layernorm = GemmaRMSNorm(
                    dim=self.config.paligemma_config.text_config.hidden_size,
                    cond_dim=paligemma_cond_dim
                )
                self.paligemma.language_model.model.layers[layer_idx].post_attention_layernorm = GemmaRMSNorm(
                    dim=self.config.paligemma_config.text_config.hidden_size,
                    cond_dim=paligemma_cond_dim
                )

                # gemma_expert
                self.gemma_expert.model.layers[layer_idx].input_layernorm = GemmaRMSNorm(
                    dim=self.config.gemma_expert_config.hidden_size,
                    cond_dim=gemma_expert_cond_dim
                )
                self.gemma_expert.model.layers[layer_idx].post_attention_layernorm = GemmaRMSNorm(
                    dim=self.config.gemma_expert_config.hidden_size,
                    cond_dim=gemma_expert_cond_dim
                )

        if self.config.add_prior:
            self.PatchEmbedDust3R_Mlp_rays = PatchEmbed_Mlp(patch_size=config.paligemma_config.vision_config.patch_size,
                                                            embed_dim=config.paligemma_config.vision_config.hidden_size,
                                                            in_chans=3)
            self.PatchEmbedDust3R_Mlp_depth = PatchEmbed_Mlp(patch_size=config.paligemma_config.vision_config.patch_size,
                                                             embed_dim=config.paligemma_config.vision_config.hidden_size,
                                                             in_chans=2)

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def _gated_residual(self, x, y, gate):
        """
        Applies gated residual connection with optional gate parameter.

        Args:
            x: Input tensor (residual)
            y: Output tensor to be added
            gate: Optional gate tensor to modulate the addition

        Returns:
            x + y if gate is None, otherwise x + y * gate
        """
        if x is None and y is None:
            return None
        if x is None or y is None:
            return x if x is not None else y
        if gate is None:
            return x + y
        return x + y * gate

    def embed_image(self, image: torch.Tensor, ray: torch.Tensor | None, depth: torch.Tensor | None):
        # image only
        if ray is None and depth is None:
            return self.paligemma.get_image_features(image)
        # +depth or +ray
        else:
            image_outputs = self.paligemma.vision_tower(image)
            selected_image_feature = image_outputs.last_hidden_state
            if ray is not None:
                ray_feature, pos = self.PatchEmbedDust3R_Mlp_rays(ray)
                selected_image_feature = selected_image_feature + ray_feature
            if depth is not None:
                depth_feature, pos = self.PatchEmbedDust3R_Mlp_depth(depth)
                selected_image_feature = selected_image_feature + depth_feature

            image_features = self.paligemma.multi_modal_projector(selected_image_feature)
            image_features = image_features / (self.paligemma.config.text_config.hidden_size ** 0.5)
            return image_features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.model.embed_tokens(tokens)

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]

        models = [self.paligemma.language_model.model, self.gemma_expert.model]
        att_vis_output = []
        prefix_emb_layer_outputs = []
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        head_dim = self.paligemma.config.text_config.head_dim
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            key_states_len = []
            value_states_len = []
            gates = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    gates.append(None)
                    continue
                layer = models[i].layers[layer_idx]
                # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
                # hidden_states = hidden_states * normalizer
                if self.pi05:
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)
                else:
                    hidden_states = layer.input_layernorm(hidden_states)


                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=torch.bfloat16)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)
                key_states_len.append(key_state.shape[1])
                value_states_len.append(value_state.shape[1])

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                    # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                    # the max len, then we (for instance) double the cache size. This implementation already exists
                    # in `transformers`. (molbap)
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )

                    # Update the cache in-place with the concatenated tensors.
                    past_key_values[layer_idx]["key_states"] = key_states
                    past_key_values[layer_idx]["value_states"] = value_states

            attention_interface = self.get_attention_interface()

            if (not self.config.is_knowledge_insulation) or (None in inputs_embeds):
                # print("Not knowledge insulation")
                att_output, probs = attention_interface(
                    attention_mask, batch_size, head_dim, query_states, key_states, value_states
                )
            else:
                # print("knowledge insulation")
                att_output, probs = self.eager_knowledge_insulation_attention_forward(
                    attention_mask, batch_size, head_dim, query_states, key_states, value_states, key_states_len
                )

            att_output = att_output.to(dtype=torch.bfloat16)  # (b, seq_vlm, 1*8*256), seq_vlm=(256*3+48)
            att_vis_output.append(probs)  # probs (b, 8, seq, seq)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # TODO: first dropout (by default 0.0)

                    # first residual
                    if self.pi05:
                        out_emb = self._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                        after_first_residual = out_emb.clone()
                        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    else:
                        out_emb += hidden_states
                        after_first_residual = out_emb.clone()
                        out_emb = layer.post_attention_layernorm(out_emb)

                    out_emb = layer.mlp(out_emb)

                    # TODO: second dropout (by default 0.0)

                    # second residual
                    if self.pi05:
                        out_emb = self._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    else:
                        out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end
                else:
                    outputs_embeds.append(None)

            prefix_emb_layer_outputs.append(outputs_embeds[0])
            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                if self.pi05:
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                else:
                    out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        # return outputs_embeds, past_key_values
        return outputs_embeds, past_key_values, att_vis_output, prefix_emb_layer_outputs

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.config.paligemma_config.text_config.num_attention_heads # 8
        num_key_value_heads = self.config.paligemma_config.text_config.num_key_value_heads # 1
        num_key_value_groups = num_att_heads // num_key_value_heads # 8

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        # print(query_states.shape, key_states.shape)
        # # q = n1,d, k = n2,d, v=n2,d
        # # attn = qk^T = n1,n2
        # # attn,v = n1,d
        # print(att_weights.shape)
        # print(attention_mask.shape)

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype) #(b, num_heads, seq_len, seq_len)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output, probs

    def eager_knowledge_insulation_attention_forward(
            self, attention_mask, batch_size, head_dim, query_states, key_states, value_states, key_states_len
    ):
        num_att_heads = self.config.paligemma_config.text_config.num_attention_heads # 8
        num_key_value_heads = self.config.paligemma_config.text_config.num_key_value_heads # 1
        num_key_value_groups = num_att_heads // num_key_value_heads # 8

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32) # (b, seq, num_att_head, head_dim)
        key_states = key_states.to(dtype=torch.float32) # (b, seq, num_att_head, head_dim)

        query_states = query_states.transpose(1, 2) # (b, num_att_head, seq, head_dim)
        key_states = key_states.transpose(1, 2) # (b, num_att_head, seq, head_dim)

        # stop gradient
        # 'b' means backbone, 'a' means action expert
        len_b = key_states_len[0]
        len_a = key_states_len[1]

        # split qkv
        query_states_b = query_states[:, :, :len_b, :]
        key_states_b = key_states[:, :, :len_b, :]
        value_states_b = value_states[:, :len_b, :, :]

        query_states_a = query_states[:, :, len_b:, :]
        key_states_a = key_states[:, :, len_b:, :]
        value_states_a = value_states[:, len_b:, :, :]

        qk_matrix_shape = (
            batch_size, num_att_heads, len_a + len_b, len_a + len_b)
        att_weights = torch.zeros(qk_matrix_shape, device=query_states.device) #(b, num_att_head, seq, seq)

        att_weights_bb = torch.matmul(query_states_b, key_states_b.transpose(2, 3))
        att_weights_ab = torch.matmul(query_states_a, key_states_b.transpose(2, 3).detach())
        att_weights_aa = torch.matmul(query_states_a, key_states_a.transpose(2, 3))

        att_weights[:, :, :len_b, :len_b] = att_weights_bb
        att_weights[:, :, len_b:, :len_b] = att_weights_ab
        att_weights[:, :, len_b:, len_b:] = att_weights_aa

        att_weights *= head_dim ** -0.5
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        probs_bb = probs[:, :, :len_b, :len_b]
        probs_ab = probs[:, :, len_b:, :len_b]
        probs_aa = probs[:, :, len_b:, len_b:]

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output_b = torch.matmul(probs_bb, value_states_b.permute(0, 2, 1, 3)) #(b, num_att_head, seq_b, head_dim)

        att_output_ab = torch.matmul(probs_ab, value_states_b.permute(0, 2, 1, 3).detach())  # (b, num_att_head, seq_a, head_dim)
        att_output_aa = torch.matmul(probs_aa, value_states_a.permute(0, 2, 1, 3)) # (b, num_att_head, seq_a, head_dim)
        att_output_a = att_output_ab + att_output_aa # (b, num_att_head, seq_a, head_dim)

        att_output = torch.cat([att_output_b, att_output_a], dim=2) # (b, num_att_head, seq_b + seq_a, head_dim)

        att_output = att_output.permute(0, 2, 1, 3) # (b, seq_b + seq_a, num_att_head, head_dim)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output, probs
