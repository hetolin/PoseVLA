import json
import os
import sys
from collections import deque
from pathlib import Path
from types import MethodType

import numpy as np
import torch


POSEVLA_ROOT = Path(__file__).resolve().parents[2]
if str(POSEVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(POSEVLA_ROOT))

from posevla.configuration_posevla import PoseVLAConfig
from posevla.modeling_posevla import PoseVLAPolicy, make_att_2d_masks


_COMPATIBLE_CONFIG_KEYS = {
    "n_obs_steps",
    "chunk_size",
    "n_action_steps",
    "max_state_dim",
    "max_action_dim",
    "resize_imgs_with_padding",
    "empty_cameras",
    "tokenizer_max_length",
    "tokenizer_model_path",
    "proj_width",
    "num_steps",
    "use_cache",
    "attention_implementation",
    "freeze_vision_encoder",
    "train_expert_only",
    "train_state_proj",
    "vis_attn",
    "is_knowledge_insulation",
    "pi05",
    "add_extra_token",
    "add_image_token",
    "add_prior",
}


class PI05Policy(PoseVLAPolicy):
    """Load the legacy PI0.5 vocabulary without resizing embeddings."""

    def resize_embeddings(self, mean_resizing=True):
        current_size = (
            self.model.paligemma_with_expert.paligemma.language_model.model
            .embed_tokens.num_embeddings
        )
        print(
            "[PI0.5 baseline] Keeping checkpoint embedding size "
            f"{current_size}; tokenizer length is {len(self.language_tokenizer)}."
        )


def _denoise_step_with_cloned_cache(
    self,
    state,
    prefix_pad_masks,
    past_key_values,
    x_t,
    timestep,
):
    """Avoid deepcopy failures on non-leaf PyTorch KV-cache tensors."""
    past_key_values_vlm = {
        layer_idx: {
            name: value.clone()
            for name, value in layer_cache.items()
        }
        for layer_idx, layer_cache in past_key_values.items()
    }

    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
        self.embed_suffix(state, x_t, timestep)
    )
    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
        batch_size,
        suffix_len,
        prefix_len,
    )
    suffix_att_2d_masks = make_att_2d_masks(
        suffix_pad_masks,
        suffix_att_masks,
    )
    full_att_2d_masks = torch.cat(
        [prefix_pad_2d_masks, suffix_att_2d_masks],
        dim=2,
    )

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    outputs_embeds, _, att_vis_output, _ = (
        self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values_vlm,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
    )
    suffix_out = outputs_embeds[1][:, -self.config.n_action_steps :]
    return self.action_out_proj(suffix_out), att_vis_output


class PI05PolicyWrapper:
    """RoboTwin wrapper for the isolated 14-dimensional PI0.5 baseline."""

    def __init__(
        self,
        ckpt_path,
        action_chunk_steps=None,
        tokenizer_model_path=None,
    ):
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as config_file:
            checkpoint_config = json.load(config_file)
        if not checkpoint_config.get("pi05", False):
            raise ValueError(f"Checkpoint is not marked as PI0.5: {config_path}")
        if checkpoint_config.get("add_extra_token", False):
            raise ValueError(
                "This baseline requires add_extra_token=false."
            )
        if checkpoint_config.get("add_prior", False):
            raise ValueError("This baseline requires add_prior=false.")

        config_kwargs = {
            key: value
            for key, value in checkpoint_config.items()
            if key in _COMPATIBLE_CONFIG_KEYS
        }
        if tokenizer_model_path:
            config_kwargs["tokenizer_model_path"] = tokenizer_model_path

        policy_config = PoseVLAConfig(**config_kwargs)
        self.policy = PI05Policy.from_pretrained(
            ckpt_path,
            config=policy_config,
            local_files_only=True,
            strict=False,
        )
        self.policy.model.denoise_step = MethodType(
            _denoise_step_with_cloned_cache,
            self.policy.model,
        )
        self.policy.cuda().eval()
        self.policy = self.policy.to(torch.bfloat16)

        self.action_dim = 14
        self.action_chunk_steps = (
            int(action_chunk_steps) if action_chunk_steps else None
        )
        self.action_cache = deque()

        print(
            f"[PI0.5 baseline] Loaded checkpoint: {ckpt_path}, "
            f"n_action_steps={self.policy.config.n_action_steps}, "
            f"action_chunk_steps={self.action_chunk_steps or 'all'}"
        )

    def reset(self):
        self.policy.reset()
        self.action_cache.clear()
        return "Model reset successful"

    def _select_action_chunk(self, batch):
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                batch[key] = torch.from_numpy(value).to(
                    device="cuda",
                    dtype=torch.bfloat16,
                )
            elif isinstance(value, torch.Tensor):
                batch[key] = value.to(
                    device="cuda",
                    dtype=torch.bfloat16,
                )

        with torch.no_grad():
            images, img_masks = self.policy.prepare_images(batch)
            state = self.policy.prepare_state(batch)
            lang_tokens, lang_masks, _ = self.policy.prepare_language(
                batch,
                is_training=False,
            )
            actions = self.policy.model.sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise=None,
                vis_attn=False,
                rays=None,
                depths=None,
            )
        return actions[:, :, : self.action_dim]

    def get_action(self, batch):
        if self.action_cache:
            return self.action_cache.popleft().cpu().numpy()

        self.policy.reset()
        action_chunk = self._select_action_chunk(batch).squeeze(0)
        if self.action_chunk_steps is not None:
            action_chunk = action_chunk[: self.action_chunk_steps]
        for action in action_chunk[1:]:
            self.action_cache.append(action)
        return action_chunk[0].cpu().numpy()


def get_model(usr_args):
    if usr_args.get("norm_path"):
        os.environ["PI05_BASELINE_NORM_PATH"] = usr_args["norm_path"]

    return PI05PolicyWrapper(
        ckpt_path=usr_args["checkpoint_path"],
        action_chunk_steps=usr_args.get("action_chunk_steps"),
        tokenizer_model_path=usr_args.get("tokenizer_model_path"),
    )


def reset_model(model):
    return model.reset()
