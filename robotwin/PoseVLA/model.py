import torch
from collections import deque
import os
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

parent_directory = str(repo_root)


def reset_model(model):
    return model.reset()


class PoseVLAPolicyWrapper:
    """RoboTwin wrapper for a PoseVLA post-trained checkpoint."""

    def __init__(self, ckpt_path, action_type="eep"):
        from posevla.modeling_posevla import PI0Policy

        self.weight_dtype = torch.bfloat16
        self.action_type = action_type
        self.action_dim = 16 if action_type == "eep" else 14

        self.policy = PI0Policy.from_pretrained(
            ckpt_path,
            local_files_only=True,
            strict=False,
        )
        self.policy.cuda()
        self.policy.eval()
        self.policy = self.policy.to(self.weight_dtype)

        self.action_cache = deque()
        self.latest_predicted_frames = None
        self.need_new_prediction = True
        self.use_mask = False

        print(
            f"[PoseVLA] Loaded checkpoint: {ckpt_path}, "
            f"action_type={self.action_type}, action_dim={self.action_dim}, "
            f"n_action_steps={self.policy.config.n_action_steps}"
        )

    def reset(self):
        self.policy.reset()
        self.action_cache.clear()
        self.need_new_prediction = True

        return "Model reset successful"

    def select_action(self, batch):
        """Run PoseVLA directly and return a full action chunk.

        We do not call the default single-step helper because its default
        action feature truncates actions to 14 dims, while Robotwin EEP uses 16 dims.
        """
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                tensor_value = torch.from_numpy(value).to(self.weight_dtype)
                batch[key] = tensor_value.cuda()
            elif isinstance(value, torch.Tensor):
                batch[key] = value.cuda().to(self.weight_dtype)

        with torch.no_grad():
            images, img_masks = self.policy.prepare_images(batch)
            state = self.policy.prepare_state(batch)
            lang_tokens, lang_masks, _ = self.policy.prepare_language(batch, is_training=False)

            results = self.policy.model.sample_actions(
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

        action = results[:, :, : self.action_dim]
        self.latest_predicted_frames = None
        return action

    def get_action(self, batch):
        """
        Wrap around policy.select_action to return a cached action chunk.
        """
        if len(self.action_cache) > 0:
            print(f"Debug - Using cached action, {len(self.action_cache)} actions remaining")
            return self.action_cache.popleft().cpu().numpy()

        self.policy.reset()
        full_actions_chunk = self.select_action(batch).squeeze(0)

        for action in full_actions_chunk[1:]:
            self.action_cache.append(action)

        return full_actions_chunk[0].cpu().numpy()


def get_model(usr_args):
    """Get a PoseVLA post-trained policy for RoboTwin evaluation."""
    if usr_args.get("norm_path"):
        os.environ["POSEVLA_NORM_PATH"] = usr_args["norm_path"]

    ckpt_path = usr_args.get("checkpoint_path")
    if not ckpt_path:
        checkpoint_id = usr_args["checkpoint_id"]
        ckpt_dir_name = usr_args.get("ckpt_dir_name", "align_bs12_1_robotwin")
        ckpt_path = os.path.join(parent_directory, f"ckpt/{ckpt_dir_name}/{checkpoint_id}/model")

    return PoseVLAPolicyWrapper(
        ckpt_path=ckpt_path,
        action_type=usr_args.get("action_type", "eep"),
    )
