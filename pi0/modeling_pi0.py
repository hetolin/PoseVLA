#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright 2026 PoseVLA Authors. Adapted for the PoseVLA project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
π0: A Vision-Language-Action Flow Model for General Robot Control.

Originally designed by Physical Intelligence and ported from JAX by Hugging Face
(`lerobot.common.policies.pi0`). This file is an adapted copy used inside the
PoseVLA project; it has been modified to support extra tokenizers, prior inputs
(rays / depths), attention-mask logic for joint VLA + VLM training, etc.

References:
- Paper: https://www.physicalintelligence.company/download/pi0.pdf
- JAX code: https://github.com/Physical-Intelligence/openpi
- HuggingFace port: https://github.com/huggingface/lerobot
"""
# ===== Standard library =====
import copy
import math
import os
import sys
from collections import deque
from pathlib import Path

# Make the project root importable when this file is executed directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# ===== Third-party libraries =====
import hydra
import omegaconf
import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer

# ===== Local modules =====
from pi0.configuration_pi0 import PI0Config
from pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from pi0._lerobot_compat import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    PreTrainedPolicy,
    get_safe_dtype,
)
from utils.mapping_token import BinTokenizer
# Debug-only helpers; referenced from commented-out call sites in this file
# (e.g. ``visualize_attention_mask(att_2d_masks)`` / ``vis_atten_map(...)``).
# Kept here so that uncommenting those lines just works without extra edits.
from utils.vis import visualize_attention_mask, vis_atten_map  # noqa: F401

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma_alpha_dist = torch.distributions.Gamma(alpha, 1)
    gamma_beta_dist = torch.distributions.Gamma(beta, 1)

    x = gamma_alpha_dist.sample((bsize,)).to(device)
    y = gamma_beta_dist.sample((bsize,)).to(device)
    z = x / (x + y)

    return z


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1, mode="bilinear"):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    interpolate_params = {
        'size': (resized_height, resized_width),
        'mode': mode,
    }

    if mode != "nearest":
        interpolate_params['align_corners'] = False

    resized_img = F.interpolate(img, **interpolate_params)


    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    # padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)

    # pad on the center of image
    pw = pad_width // 2
    ph = pad_height // 2
    padded_img = F.pad(resized_img, (pw, pad_width - pw, ph, pad_height - ph), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector

_POSEVLA_ROOT = Path(__file__).resolve().parents[1]
cfg = omegaconf.OmegaConf.load(_POSEVLA_ROOT / "config" / "base.yaml")
if not os.path.isabs(cfg.model.tokenizer_model_path):
    cfg.model.tokenizer_model_path = str(_POSEVLA_ROOT / cfg.model.tokenizer_model_path)
bin_tokenizer = BinTokenizer(cfg.statistics_path_6d_dataset)
EXTRA_TOKENS = bin_tokenizer.EXTRA_LOC_TOKENS + bin_tokenizer.EXTRA_SEG_TOKENS
EXTRA_3D_TOKENS = (bin_tokenizer.EXTRA_TRANS_XY_TOKENS +
                   bin_tokenizer.EXTRA_TRANS_Z_TOKENS +
                   bin_tokenizer.EXTRA_ROT_XYZ_TOKENS +
                   bin_tokenizer.EXTRA_SIZE_XYZ_TOKENS)
EXTRA_IMAGE_TOKENS = bin_tokenizer.EXTRA_IMAGE_TOKENS
EXTRA_NO_OBJ_TOKENS = bin_tokenizer.EXTRA_NO_OBJ_TOKENS


def _resolve_posevla_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    local_path = _POSEVLA_ROOT / path
    if local_path.exists():
        return str(local_path)
    return path


class PI0Policy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        config.tokenizer_model_path = _resolve_posevla_path(config.tokenizer_model_path)
        self.language_tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_model_path,
            local_files_only=True,
        )

        if config.add_extra_token:
            self.language_tokenizer.add_tokens(EXTRA_TOKENS)
            self.language_tokenizer.add_tokens(EXTRA_3D_TOKENS)
            self.language_tokenizer.add_tokens(EXTRA_IMAGE_TOKENS)
            self.language_tokenizer.add_tokens(EXTRA_NO_OBJ_TOKENS)
            print("Add extra token")
            print("len of new language_tokenizer = ", len(self.language_tokenizer))

        self.model = PI0FlowMatching(config, self.language_tokenizer)

        self.reset()

        # Auto-resize embeddings if the tokenizer vocab exceeds the current
        # embedding size. The check is idempotent:
        # - `from_pretrained` inference: the checkpoint already stores the
        #   resized embedding, so tokenizer == embed_size and we skip.
        # - First-time training: tokenizer > original embed_size, so we
        #   trigger an auto-resize with `mean_resizing=True`.
        # - When loading raw VLM weights, prefer `load_pretrained_vlm()`,
        #   which performs the resize in the correct order.
        current_embed_size = self.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.num_embeddings
        if len(self.language_tokenizer) > current_embed_size:
            print(
                f"[Auto Resize] Tokenizer vocab ({len(self.language_tokenizer)}) > "
                f"embedding size ({current_embed_size}), auto resizing..."
            )
            self.resize_embeddings(mean_resizing=True)

    def resize_embeddings(self, mean_resizing=True):
        # Idempotent guard: skip when the embedding already matches the
        # tokenizer length, otherwise repeated resizes would corrupt weights.
        current_embed_size = self.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.num_embeddings
        target_size = len(self.language_tokenizer)
        if current_embed_size == target_size:
            print(f"[resize_embeddings] Skipped: embedding size already matches tokenizer ({target_size}).")
            return

        print("+++++++++++++++++++++++ Resizing Embeddings ++++++++++++++++++++++++")
        # `resize_token_embeddings` expects the FULL size of the new vocabulary
        # (i.e. `len(tokenizer)`), e.g. token: 0-257151, special_token:
        # 257152 <image>, etc.
        print(f"using mean_resizing={mean_resizing}")
        print(f"current embedding size = {current_embed_size}")
        self.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens = (
            self.model.paligemma_with_expert.paligemma.resize_token_embeddings(
                target_size, mean_resizing=mean_resizing
            )
        )
        print(
            f"Resized embedding size = "
            f"{self.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.num_embeddings}"
        )

    def load_pretrained_vlm(
        self,
        pretrained_vlm_path: str,
        mean_resizing: bool = True,
        local_files_only: bool = True,
    ):
        """Overwrite the paligemma submodule with a raw pretrained VLM
        checkpoint and resize embeddings in the correct order.

        Typical usage:
            policy = PI0Policy.from_pretrained(pi0_ckpt, config=pi0_config, strict=False)  # action expert
            policy.load_pretrained_vlm("pretrain/paligemma-3b-pt-224")                     # overwrite paligemma

        Correct ordering:
            1. If `__init__` already auto-resized embeddings, shrink them
               back to the original vocab so the shapes match the
               pretrained weights.
            2. `load_state_dict` to load the pretrained weights.
            3. `tie_weights()` to re-bind `lm_head.weight` to
               `embed_tokens.weight` (when `tie_word_embeddings=True`,
               `lm_head.weight` is not stored in the checkpoint and must
               be re-tied).
            4. Resize back to `len(tokenizer)`, performing mean_resizing
               on top of the freshly loaded weights.
        """
        from transformers import PaliGemmaForConditionalGeneration

        # Step 1: Load the pretrained paligemma state_dict.
        pretrained_model = PaliGemmaForConditionalGeneration.from_pretrained(
            pretrained_vlm_path, local_files_only=local_files_only
        )
        pretrained_state = pretrained_model.state_dict()
        del pretrained_model

        # Probe the embedding size of the pretrained weights (key names
        # vary across transformers versions).
        embed_key_candidates = [
            "language_model.model.embed_tokens.weight",
            "model.language_model.model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        ]
        pretrained_embed_size = None
        for k in embed_key_candidates:
            if k in pretrained_state:
                pretrained_embed_size = pretrained_state[k].shape[0]
                break
        if pretrained_embed_size is None:
            raise RuntimeError(
                f"[load_pretrained_vlm] Cannot locate embed_tokens.weight in pretrained state_dict. "
                f"Available top-level keys (first 10): {list(pretrained_state.keys())[:10]}"
            )

        current_embed_size = self.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.num_embeddings

        # Step 2: If the embedding has already been resized, shrink it
        # back to the original size so it matches the pretrained weights.
        if current_embed_size != pretrained_embed_size:
            print(
                f"[load_pretrained_vlm] Shrinking embedding {current_embed_size} -> "
                f"{pretrained_embed_size} to match pretrained weights..."
            )
            self.model.paligemma_with_expert.paligemma.resize_token_embeddings(pretrained_embed_size)

        # Step 3: Load the pretrained weights (shapes now match exactly).
        missing, unexpected = self.model.paligemma_with_expert.paligemma.load_state_dict(
            pretrained_state, strict=False
        )
        del pretrained_state
        print(f"[load_pretrained_vlm] Loaded pretrained VLM from: {pretrained_vlm_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

        # Step 4: Explicit tie_weights so that `lm_head.weight` points to
        # `embed_tokens.weight`.
        self.model.paligemma_with_expert.paligemma.tie_weights()
        print("[load_pretrained_vlm] tie_weights() called, lm_head.weight is tied to embed_tokens.weight")

        # Step 5: Mean-resize back to the tokenizer length on top of the
        # freshly loaded weights.
        if len(self.language_tokenizer) > pretrained_embed_size:
            print(
                f"[load_pretrained_vlm] Resizing embedding {pretrained_embed_size} -> "
                f"{len(self.language_tokenizer)} with mean_resizing={mean_resizing}..."
            )
            self.resize_embeddings(mean_resizing=mean_resizing)

        return missing, unexpected

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            rays = None
            depths = None
            if self.config.add_prior:
                rays = self.prepare_rays(batch)
                depths = self.prepare_depths(batch)

            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks, lang_token_type_ids = self.prepare_language(batch)
            results = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, vis_attn=self.config.vis_attn, rays=rays, depths=depths,
            )
            if self.config.vis_attn:
                actions, _ = results
            else:
                actions = results

            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Run a full training forward pass and compute the loss."""
        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks, lang_token_type_ids = self.prepare_language(batch)

        # TODO: currently only VLM data is supported for prior inputs.
        rays = None
        depths = None
        if self.config.add_prior:
            rays = self.prepare_rays(batch)
            depths = self.prepare_depths(batch)

        if batch.get("action") is not None:
            state = self.prepare_state(batch)
            actions = self.prepare_action(batch)
            actions_is_pad = batch.get("actions_is_pad")
        else:
            state = None
            actions = None
            actions_is_pad = None

        if batch.get("text_label") is not None:
            lang_token_labels = lang_tokens.masked_fill(
                lang_token_type_ids == self.model.paligemma_with_expert.paligemma.config.pad_token_id,
                self.model.paligemma_with_expert.paligemma.config.ignore_index,
            )
        else:
            lang_token_labels = None

        loss_dict = {}
        losses_flow, losses_ntp = self.model.forward(
            images, img_masks, lang_tokens, lang_masks,
            state, actions, noise, time,
            lang_token_labels, rays, depths,
        )
        loss_flow = 0
        loss_ntp = 0

        if losses_flow is not None:
            loss_dict["losses_after_forward"] = losses_flow.clone()

            if actions_is_pad is not None:
                in_episode_bound = ~actions_is_pad
                losses_flow = losses_flow * in_episode_bound.unsqueeze(-1)
                loss_dict["losses_after_in_ep_bound"] = losses_flow.clone()

            # Remove padding
            losses_flow = losses_flow[:, :, : self.config.max_action_dim]
            loss_dict["losses_after_rm_padding"] = losses_flow.clone()

            loss_flow = losses_flow.mean()
            # For logging
            loss_dict["flow_loss"] = loss_flow.item()

        if losses_ntp is not None:
            loss_ntp = losses_ntp.mean()
            # For logging
            loss_dict["ntp_loss"] = loss_ntp.item()

        # For backward pass
        loss_dict["loss"] = loss_flow + loss_ntp

        return loss_dict

    @torch.no_grad()
    def forward_evaluate(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks, _ = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_is_pad")

        rays = None
        depths = None
        if self.config.add_prior:
            rays = self.prepare_rays(batch)
            depths = self.prepare_depths(batch)

        results = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=None, vis_attn=self.config.vis_attn, rays=rays, depths=depths,
        )
        if self.config.vis_attn:
            pred_actions, att_vis_output = results
            info = {"pred": pred_actions, "gt": actions, "attn": att_vis_output}
        else:
            pred_actions = results
            info = {"pred": pred_actions, "gt": actions}

        return info

    @torch.no_grad()
    def forward_evaluate_ntp(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run a forward pass that produces next-token predictions."""
        # TODO: only batch size 1 is supported.
        for k in batch:
            batch[k] = batch[k][0:1]
        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks, _ = self.prepare_language(batch, is_training=False)

        # TODO: currently only VLM data is supported for prior inputs.
        rays = None
        depths = None
        if self.config.add_prior:
            rays = self.prepare_rays(batch)
            depths = self.prepare_depths(batch)

        pred_token_ids = self.model.next_token_predict(
            images, img_masks, lang_tokens, lang_masks,
            stop_token=self.language_tokenizer.eos_token_id,
            max_tokens_to_generate=self.config.tokenizer_max_length,
            rays=rays, depths=depths,
        )
        # ``next_token_predict`` always returns (B, T); decode the single
        # row corresponding to this sample.
        pred_texts = self.language_tokenizer.decode(pred_token_ids[0])

        info = {
            "pred": [pred_texts],
            "gt": batch["text_label"],
        }

        return info


    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                bs, history_len, c, h, w = img.shape
                img = img.reshape(bs * history_len, c, h, w)
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                # img = img.reshape(bs, history_len, c, h, w)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch, is_training=True) -> tuple[Tensor, Tensor]:
        """Tokenize the text input."""
        device = next(v.device for k, v in batch.items() if k.startswith(OBS_IMAGES))
        tasks = batch["task"]

        # Clean text
        tasks = [task.strip().replace("_", " ").replace("\n", " ") for task in tasks]

        state = batch.get(OBS_STATE)
        if self.config.pi05 and state is not None:
            state = batch[OBS_STATE]
            assert len(state.shape) == 2, f"state shape should be (B, N), but got {state.shape}"

            # Pi05 format: state is part of the discrete language input.
            # Use 256 bins; equivalent numpy: bins = np.linspace(-1, 1, 257)[:-1]
            bins = torch.linspace(-1, 1, 256 + 1)[:-1]  # shape: (256,)
            bins = bins.to(state.dtype).to(state.device)
            # Note: torch.bucketize defines `right` differently from
            # np.digitize. We pass right=True here to align with the
            # np.digitize behavior used in openpi.
            discretized_state = torch.bucketize(state, bins, right=True) - 1  # range: [0, 255]
            # Each element becomes a token; e.g. " -" is treated as a single token.
            state_strs = [" ".join(map(str, row.tolist())) for row in discretized_state]

            tasks = [
                f"Task: {task.strip()}, State: {state_str};\nAction: "
                for task, state_str in zip(tasks, state_strs)
            ]
        else:
            # PaliGemma prompt has to end with a new line.
            # In Pi0 format the state is part of the continuous action expert
            # input; here we tokenize "\n" separately as the
            # "start of answer" token.
            tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        task_labels = batch.get("text_label")
        if task_labels is not None:
            tasks = [self.language_tokenizer.bos_token + task for task in tasks]
            task_labels = [task_label + self.language_tokenizer.eos_token for task_label in task_labels]
        else:
            tasks = [self.language_tokenizer.bos_token + task for task in tasks]

        if is_training:
            tokenized_prompt = self.language_tokenizer.__call__(
                tasks,
                text_pair=task_labels,
                padding="max_length",
                # padding="longest",
                padding_side="right",
                truncation=True,
                max_length=self.config.tokenizer_max_length,
                return_tensors="pt",
                add_special_tokens=False,
                return_token_type_ids=True,
            )
        else:
            tokenized_prompt = self.language_tokenizer.__call__(
                tasks,
                padding="longest",
                padding_side="right",
                truncation=True,
                max_length=self.config.tokenizer_max_length,
                return_tensors="pt",
                add_special_tokens=False,
                return_token_type_ids=True,
            )

        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)
        lang_token_type_ids = tokenized_prompt["token_type_ids"].to(device=device)

        return lang_tokens, lang_masks, lang_token_type_ids

    def prepare_rays(self, batch):
        images = []
        img_masks = []

        ray_features = ['observation.rays.top_head', 'observation.rays.hand_left', 'observation.rays.hand_right']
        present_img_keys = [key for key in ray_features if key in batch]
        missing_img_keys = [key for key in ray_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            bs, history_len, c, h, w = img.shape
            img = img.reshape(bs * history_len, c, h, w)

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * 0
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images

    def prepare_depths(self, batch):
        images = []
        img_masks = []

        det_features = ['observation.depth_priors.top_head', 'observation.depth_priors.hand_left', 'observation.depth_priors.hand_right']
        present_img_keys = [key for key in det_features if key in batch]
        missing_img_keys = [key for key in det_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            bs, history_len, c, h, w = img.shape
            img = img.reshape(bs * history_len, c, h, w)

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * 0
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


class PI0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config, language_tokenizer):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.language_tokenizer = language_tokenizer

        paligemma_with_expert_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            is_knowledge_insulation=self.config.is_knowledge_insulation,
            pi05=self.config.pi05,
            use_adarms=[False, True] if self.config.pi05 else [False, False],
            add_prior=self.config.add_prior,
            skip_init_weights=getattr(self.config, "skip_init_weights", False),
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_expert_config)

        # Projections are float32
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(self.config.proj_width, self.config.proj_width)
            self.time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        else:
            self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
            self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
            self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

            self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def build_attention_mask_from_tokens(self, lang_tokens, start_token_id=2, sep_token_id=108):
        """
        Build an attention mask where tokens between the first start_token_id
        and the first sep_token_id (inclusive) are 0, and tokens after that
        are 1.

        Args:
            tokens: token-id tensor of shape [batch_size, seq_len].
            start_token_id: id of the start marker (default 2, usually BOS).
            end_token_id: id of the end/separator marker
                (default 108, usually a question mark / special token).

        Returns:
            Attention-mask tensor of shape [batch_size, seq_len], values 0/1.
        """
        batch_size, seq_len = lang_tokens.shape
        device = lang_tokens.device

        # Find the first start_token_id / end_token_id position per row
        start_pos = (lang_tokens == start_token_id).float().argmax(dim=1)  # [batch_size]
        # end_token_id must be located after start_pos, so mask out anything before start
        idxs = torch.arange(seq_len, device=device)[None, :]  # [1, seq_len]
        # Only sep_tokens occurring at/after start_pos are valid
        valid_end_mask = (lang_tokens == sep_token_id) & (idxs >= start_pos[:, None])
        # Find end_pos; if none exists, fall back to the last position
        end_exists = valid_end_mask.any(dim=1)
        end_pos = torch.where(end_exists,
                              valid_end_mask.float().argmax(dim=1),
                              torch.full_like(start_pos, seq_len - 1)
                              )

        # Build the mask: 1 after end_pos, 0 elsewhere
        att_masks = (idxs > end_pos[:, None]).int()

        return att_masks



    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, rays=None, depths=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        rays = rays if rays is not None else [None] * len(images)
        depths = depths if depths is not None else [None] * len(images)
        # TODO: remove for loop

        for i, (img, img_mask, ray, depth) in enumerate(zip(images, img_masks, rays, depths, strict=True)):
            num_start_end_embs = 2 if self.config.add_image_token else 0
            if self.config.add_image_token:
                # add <imagen>
                vision_start_tokens = torch.full((img.shape[0], 1),
                                                 self.language_tokenizer.convert_tokens_to_ids(f"<image{i:0>1}>"))
                vision_start_tokens = vision_start_tokens.to(img.device)
                # add </imagen>
                vision_end_tokens = torch.full((img.shape[0], 1),
                                               self.language_tokenizer.convert_tokens_to_ids(f"</image{i:0>1}>"))
                vision_end_tokens = vision_end_tokens.to(img.device)

                embs.append(self.paligemma_with_expert.embed_language_tokens(vision_start_tokens))

            img_emb = self.paligemma_with_expert.embed_image(img, ray, depth)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            # img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            img_mask = img_mask[:, None].expand(bsize, num_img_embs + num_start_end_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            if self.config.add_image_token:
                embs.append(self.paligemma_with_expert.embed_language_tokens(vision_end_tokens))

            # Create attention masks so that image tokens attend to each other
            # att_masks += [0] * num_img_embs
            att_masks += [0] * (num_img_embs + num_start_end_embs)

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        # num_lang_embs = lang_emb.shape[1]
        # att_masks += [0] * num_lang_embs
        # create lang mask
        att_masks_lang = self.build_attention_mask_from_tokens(lang_tokens)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        # create image mask
        att_masks_img = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks_img = att_masks_img[None, :].expand(bsize, len(att_masks_img))

        # att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        # att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        att_masks = torch.cat([att_masks_img, att_masks_lang], dim=1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        if not self.pi05:
            state_emb = self.state_proj(state)
            state_emb = state_emb.to(dtype=torch.bfloat16)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            dtype = state_emb.dtype
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=state.device
        )
        time_emb = time_emb.type(dtype=torch.bfloat16)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions.to(torch.bfloat16))  # torch.float32 -> bf16

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # torch.float32

            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)  # swish == silu
            action_time_emb = self.action_time_mlp_out(action_time_emb)

            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb) # swish == silu
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)

            action_time_emb = action_emb
            adarms_cond = time_emb


        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=state.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1) # torch.bfloat16
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state=None,
        actions=None,
        noise=None,
        time=None,
        lang_token_labels=None,
        rays=None,
        depths=None,
    ) -> Tensor:
        """Run a full training forward pass and compute the loss
        of shape (batch_size, num_steps, num_motors)."""
        losses_flow = None
        losses_ntp = None

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, rays, depths
        )

        # action, text + action
        if actions is not None:
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)

            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # text only
        else:
            suffix_embs = None
            adarms_cond = None
            pad_masks = torch.cat([prefix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # visualize_attention_mask(att_2d_masks)
        (prefix_out, suffix_out), _, att_vis_output, _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond]
        )

        # Flow matching prediction
        if actions is not None:
            suffix_out = suffix_out[:, -self.config.n_action_steps:]
            # Original openpi code, upcast attention output
            # suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)  # torch.float32 -> bf16
            losses_flow = F.mse_loss(u_t.float(), v_t.float(), reduction="none")  # bf16 -> torch.float32

        # Next-token prediction
        if lang_token_labels is not None:
            attention_mask = None
            logits = self.paligemma_with_expert.paligemma.language_model.lm_head(prefix_out)

            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., -self.config.tokenizer_max_length:-1, :]
            shift_labels = lang_token_labels[..., 1:]

            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()

            # Flatten the tokens
            losses_ce = nn.CrossEntropyLoss(
                reduction="none",
                ignore_index=self.paligemma_with_expert.paligemma.config.ignore_index,
            )
            flat_logits = shift_logits.view(-1, self.paligemma_with_expert.paligemma.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            losses_ntp = losses_ce(flat_logits, flat_labels)

        return losses_flow, losses_ntp

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, vis_attn=False, rays=None, depths=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, rays, depths
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        (prefix_out, suffix_out), past_key_values, _, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t, att_vis_output = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        if vis_attn:
            # vis_atten_map(att_vis_output, images)
            return x_t, att_vis_output

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # IMPORTANT: copy the past_key_values, or its size will increase during n-step denoise.
        past_key_values_vlm = copy.deepcopy(past_key_values)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _, att_vis_output, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values_vlm,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond]
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out) # bf16 -> torch.float32
        return v_t, att_vis_output

    def next_token_predict(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        stop_token: int = 1,
        max_tokens_to_generate: int = 256,
        rays=None,
        depths=None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> Tensor:
        """Greedy / top-k / top-p decoding that handles arbitrary batch sizes.

        Returns a tensor of shape ``(B, T)`` with the generated token ids
        (``T`` <= ``max_tokens_to_generate``). Already-finished samples are
        padded with ``stop_token`` so that all rows have the same length.

        Note: ``batch_size == 1`` is just a degenerate case of this routine,
        so the previously separate ``next_token_predict_batch`` has been
        merged in here.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, rays, depths
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)  # (B, N, N)

        batch_size = prefix_pad_masks.size(0)
        device = prefix_pad_masks.device

        # Position of the last *valid* (non-pad) token of the full prefix
        # (image tokens + language tokens) for each sample. This is the
        # position responsible for predicting the very first generated token,
        # and it is robust against padding-side / per-sample length
        # differences. Using ``logits[:, -1, :]`` would silently read a pad
        # position when ``padding_side="right"`` and lengths differ, which
        # was the source of the garbled outputs from the old batched path.
        last_valid_pos = prefix_pad_masks.sum(dim=1).long() - 1  # (B,)

        generated_tokens: list[Tensor] = []
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_tokens_to_generate):
            fill_kv_cache = (i == 0)

            if past_key_values is not None:
                # Only the freshly-appended token is fed in; its position is
                # ``cumsum(pad_mask)[-1] - 1``.
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1)[:, -1] - 1  # (B,)
                prefix_position_ids = prefix_position_ids.unsqueeze(-1)  # (B, 1)
            else:
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1  # (B, N)

            (prefix_output, _), past_key_values, _, _ = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                fill_kv_cache=fill_kv_cache,
            )
            logits = self.paligemma_with_expert.paligemma.language_model.lm_head(prefix_output)

            # Step 0: gather logits at the *last valid* prefix position for
            # each sample. Step >0 only feeds in the newly-generated token,
            # so its logits are at index -1.
            if i == 0:
                next_token_logits = logits[torch.arange(batch_size, device=device), last_valid_pos, :]
            else:
                next_token_logits = logits[:, -1, :]

            # ===== Sampling =====
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            next_token_logits = filter_logits_top_k(next_token_logits, top_k)
            next_token_logits = filter_logits_top_p(next_token_logits, top_p)

            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)  # (B,)

            # Force already-finished rows to keep emitting ``stop_token``.
            next_token = torch.where(
                finished,
                torch.full_like(next_token, stop_token),
                next_token,
            )

            generated_tokens.append(next_token.unsqueeze(1))  # (B, 1)

            finished = finished | (next_token == stop_token)
            if finished.all():
                break

            # Append the new token: extend pad mask / attention mask / embeds.
            input_ids = next_token.unsqueeze(-1)  # (B, 1)
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_attn_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, 1, prefix_len)
            last_token_pad_masks = torch.ones(
                batch_size, 1, dtype=prefix_pad_masks.dtype, device=device
            ).bool()
            last_token_att_masks = torch.ones(
                batch_size, 1, dtype=prefix_att_masks.dtype, device=device
            )
            last_token_att_2d_masks = make_att_2d_masks(last_token_pad_masks, last_token_att_masks)
            prefix_att_2d_masks = torch.cat([prefix_pad_attn_2d_masks, last_token_att_2d_masks], dim=2)
            prefix_pad_masks = torch.cat([prefix_pad_masks, last_token_pad_masks], dim=1)

            lang_emb = self.paligemma_with_expert.embed_language_tokens(input_ids)
            lang_emb_dim = lang_emb.shape[-1]
            lang_emb = lang_emb * math.sqrt(lang_emb_dim)
            prefix_embs = lang_emb  # only the new token is fed in next step

        # (B, T)
        generated_token_ids = torch.cat(generated_tokens, dim=1)
        return generated_token_ids

def filter_logits_top_k(logits, k):
    if k <= 0:
        return logits
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    mask = torch.full_like(logits, True, dtype=torch.bool)
    mask.scatter_(1, top_k_indices, False)
    logits = logits.masked_fill(mask, float('-inf'))
    return logits

def filter_logits_top_p(logits, p):
    if p >= 1.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 0] = False  # always keep at least the most probable token
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask, float('-inf'))
    return logits


@hydra.main(
        version_base=None,
        config_path="../config",
        config_name="base",
    )
def test(cfg: DictConfig):
    """End-to-end smoke test that exercises all four detection datasets
    (omni6d / omni3d / bop / clutter) through the shared ``_build_vlm_3d``
    factory in ``data/factory.py``.
    """
    from data.factory import _build_vlm_3d, build_action_dataloader

    weight_dtype = torch.bfloat16
    device = "cuda:0"
    pi0_config = PI0Config(
        empty_cameras=0,
        freeze_vision_encoder=False,
        is_knowledge_insulation=False,
        pi05=False,
        vis_attn=True,
        add_extra_token=True,
        add_image_token=True,
        add_prior=True,
        skip_init_weights=True
    )

    # (A) Load from a full lerobot_pi0 checkpoint (most common; current
    #     train_pretrain.py default).
    # policy = PI0Policy.from_pretrained(cfg.model.pretrained_model_path, config=pi0_config, strict=False)

    # (B) Use only the paligemma VLM and build the action expert from scratch.
    policy = PI0Policy(pi0_config)
    policy.load_pretrained_vlm("pretrain/paligemma-3b-pt-224")

    # (C) Action expert from pi0 + overwrite paligemma with raw VLM weights.
    # policy = PI0Policy.from_pretrained(pi0_ckpt, config=pi0_config, strict=False)
    # policy.load_pretrained_vlm("pretrain/paligemma-3b-pt-224")

    policy.to(weight_dtype).to(device)

    # ---------------------------------------------------------------- #
    # 1) Action (VLA) dataloader -- single hdf5 dataset for sanity.
    # ---------------------------------------------------------------- #
    cfg.dataset.dataset_list = [cfg.dataset.xtrainer]
    train_dataloader, _ = build_action_dataloader(cfg, pi0_config)
    batch = next(iter(train_dataloader))
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(weight_dtype).to(device)

    # ---------------------------------------------------------------- #
    # 2) VLM (3D detection) dataloaders -- omni6d / omni3d / bop / clutter
    #    are all wired up inside ``_build_vlm_3d``.
    # ---------------------------------------------------------------- #
    cfg.training.weighted_sample = True
    train_vlm_dataloader, val_vlm_dataloader, train_vlm_dataset = _build_vlm_3d(
        cfg, bin_tokenizer
    )
    print(f"[test] VLM concat dataset size = {len(train_vlm_dataset)}")

    def _to_device(b):
        for k in b:
            if isinstance(b[k], torch.Tensor):
                b[k] = b[k].to(weight_dtype).to(device)
        return b

    print("========= vlm train batch ==========")
    batch_vlm_train = _to_device(next(iter(train_vlm_dataloader)))

    print("========= vlm val batch ==========")
    batch_vlm_val = _to_device(next(iter(val_vlm_dataloader)))

    # ---------------------------------------------------------------- #
    # 3) Forward smoke tests.
    # ---------------------------------------------------------------- #
    with torch.no_grad():
        # ---- VLA (action) forward -- kept commented out by default ----
        # print("========= vla forward ==========")
        # output_dict = policy.forward(batch)
        # print("vla loss =", output_dict["loss"])
        # print("========= vla forward_evaluate ==========")
        # _ = policy.forward_evaluate(batch)
        # _ = policy.select_action(batch)

        print("========= vlm train: forward ==========")
        loss_train = policy.forward(batch_vlm_train)["loss"]
        print("vlm train loss =", loss_train)

        print("========= vlm train: forward_evaluate_ntp ==========")
        _ = policy.forward_evaluate_ntp(batch_vlm_train)

        print("========= vlm val: forward ==========")
        loss_val = policy.forward(batch_vlm_val)["loss"]
        print("vlm val loss =", loss_val)

        print("========= vlm val: forward_evaluate_ntp ==========")
        _ = policy.forward_evaluate_ntp(batch_vlm_val)

    print("[test] all four detection datasets passed forward / NTP eval ✅")


if __name__ == '__main__':
    test()




