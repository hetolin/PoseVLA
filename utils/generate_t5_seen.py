#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate episode-level T5 embeddings for `instructions.json["seen"]`.

This script is designed for DROID / RoboInter-style processed data, where each
episode directory looks like:

    task_xxxxx__task-name/
      episode_dir/
        episode.hdf5
        cam_high.mp4
        segmentation_primary.npz
        instructions.json
        meta.json

For every episode directory that contains `instructions.json`, this script
encodes the `seen` instruction list and saves a single `t5_seen.pt` next to it.
The saved format is:

    List[Tensor[L_i, 4096]]

where each list item corresponds to one entry in `instructions.json["seen"]`
in the exact same order.

Examples:
    python generate_t5_seen.py \
      --dataset_roots /home/hanyang/code/wam/mydata/roboinfer_droid \
      --wan_path /path/to/pretrained_models \
      --devices 0,1
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
BAK_ROOT = REPO_ROOT / "Motus" / "bak"
if str(BAK_ROOT) not in sys.path:
    sys.path.insert(0, str(BAK_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOTS = [
    "/home/tione/notebook/workspace/hanyangyu/RoboTwin/process_data_500" # "/home/tione/notebook/workspace/hetolin/datasets/converted_robointer/RoboInterTools/converted_xtrainer/mask_prompt_xyz"  #"/home/tione/notebook/workspace/hetolin/datasets/converted_robointer/robointer_droid"
]
DEFAULT_OUTPUT_NAME = "t5_seen.pt"
DEFAULT_TEXT_LEN = 512
# DEFAULT_META_PREFIX = (
#     "The whole scene is in a realistic, industrial art style. "
#     "The robot is currently performing the following task: "
# )
WAN_REPO_PATH = "pretrained_models/Wan2.2-TI2V-5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate episode-level T5 embeddings for instructions.json['seen']"
    )
    parser.add_argument(
        "--dataset_roots",
        nargs="+",
        default=DEFAULT_DATASET_ROOTS,
        help="One or more dataset roots to recursively scan for episode directories",
    )
    parser.add_argument(
        "--wan_path",
        type=str,
        default=WAN_REPO_PATH,
        help=(
            "Path to WAN pretrained models root or directly to the Wan2.2-TI2V-5B "
            "directory."
        ),
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA device ids (for example: 0,1,2). "
            "Use 'cpu' to force CPU. Defaults to all visible CUDA devices or CPU."
        ),
    )
    return parser.parse_args()


def resolve_wan_model_dir(wan_path: str) -> Path:
    base = Path(wan_path).expanduser().resolve()
    direct_ckpt = base / "models_t5_umt5-xxl-enc-bf16.pth"
    direct_tokenizer = base / "google" / "umt5-xxl"
    nested = base / "Wan2.2-TI2V-5B"
    nested_ckpt = nested / "models_t5_umt5-xxl-enc-bf16.pth"
    nested_tokenizer = nested / "google" / "umt5-xxl"

    if direct_ckpt.exists() and direct_tokenizer.exists():
        return base
    if nested_ckpt.exists() and nested_tokenizer.exists():
        return nested

    raise FileNotFoundError(
        "Could not resolve WAN T5 model directory from "
        f"{base}. Expected either `models_t5_umt5-xxl-enc-bf16.pth` directly under it, "
        "or under `Wan2.2-TI2V-5B/`."
    )


def parse_device_list(devices_arg: str | None) -> List[str]:
    if devices_arg is not None:
        raw = [part.strip() for part in devices_arg.split(",") if part.strip()]
        if not raw:
            return ["cpu"]
        if len(raw) == 1 and raw[0].lower() == "cpu":
            return ["cpu"]
        return [f"cuda:{device_id}" for device_id in raw]

    try:
        import torch
    except ModuleNotFoundError:
        return ["cpu"]

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    return ["cpu"]


def normalize_seen_list(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen = data.get("seen", [])
    normalized = [str(item).strip() for item in seen if str(item).strip()]
    return normalized


def collect_episode_pairs(dataset_roots: Sequence[str]) -> Tuple[List[Tuple[str, str]], int, int]:
    pairs: List[Tuple[str, str]] = []
    skipped_existing = 0
    skipped_invalid = 0
    visited = set()

    for root in dataset_roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            logger.warning(f"Dataset root not found, skipping: {root_path}")
            continue

        logger.info(f"Scanning dataset root: {root_path}")
        for json_path in sorted(root_path.rglob("instructions.json")):
            episode_dir = json_path.parent
            episode_key = str(episode_dir.resolve())
            if episode_key in visited:
                continue
            visited.add(episode_key)

            has_episode_hdf5 = (
                (episode_dir / "episode.hdf5").exists()
                or (episode_dir / "episode.HDF5").exists()
                or any(episode_dir.glob("episode_*.hdf5"))
                or any(episode_dir.glob("episode_*.HDF5"))
            )
            if not has_episode_hdf5:
                continue

            try:
                seen = normalize_seen_list(str(json_path))
            except Exception as exc:
                logger.warning(f"Failed to parse {json_path}: {exc}")
                skipped_invalid += 1
                continue

            if not seen:
                logger.warning(f"Skipping {json_path}: empty or missing 'seen' list")
                skipped_invalid += 1
                continue

            output_path = episode_dir / DEFAULT_OUTPUT_NAME
            if output_path.exists() and output_path.stat().st_size > 0:
                skipped_existing += 1
                continue

            pairs.append((str(json_path), str(output_path)))

    return pairs, skipped_existing, skipped_invalid


class T5EmbeddingProcessor:
    """Encodes a `seen` instruction list into a single episode-level `.pt` file."""

    def __init__(self, wan_model_dir: str, text_len: int = 512, device: str = "cuda:0"):
        self.wan_model_dir = Path(wan_model_dir)
        self.text_len = text_len
        self.device = device
        self._encoder = None

    def _init_encoder(self):
        if self._encoder is not None:
            return

        import torch
        from wan.modules.t5 import T5EncoderModel

        device_obj = torch.device(self.device)
        dtype = torch.bfloat16 if device_obj.type == "cuda" else torch.float32

        if device_obj.type == "cuda":
            torch.cuda.set_device(device_obj)

        self._encoder = T5EncoderModel(
            text_len=self.text_len,
            dtype=dtype,
            device=device_obj,
            checkpoint_path=str(self.wan_model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(self.wan_model_dir / "google" / "umt5-xxl"),
        )
        logger.info(f"T5 encoder initialised on {self.device}")

    def process_seen_instructions(self, json_path: str, output_path: str) -> bool:
        try:
            import torch

            seen = normalize_seen_list(json_path)
            if not seen:
                logger.warning(f"No usable 'seen' instructions in {json_path}")
                return False

            output_file = Path(output_path)
            if output_file.exists() and output_file.stat().st_size > 0:
                logger.info(f"Skipping existing file: {output_file}")
                return True

            # prompts = [f"{DEFAULT_META_PREFIX}{instruction}" for instruction in seen]
            prompts = [f"{instruction}" for instruction in seen]

            self._init_encoder()
            device_obj = torch.device(self.device)
            encoded = self._encoder(prompts, device_obj)

            if isinstance(encoded, list):
                encoded_list = [emb.detach().cpu() if torch.is_tensor(emb) else torch.from_numpy(emb) for emb in encoded]
            elif torch.is_tensor(encoded):
                if encoded.ndim == 3:
                    encoded_list = [emb.detach().cpu() for emb in encoded]
                elif encoded.ndim == 2:
                    encoded_list = [encoded.detach().cpu()]
                else:
                    raise ValueError(f"Unexpected tensor output shape: {tuple(encoded.shape)}")
            else:
                raise TypeError(f"Unexpected encoder output type: {type(encoded)}")

            if len(encoded_list) != len(seen):
                raise ValueError(
                    f"Embedding count mismatch for {json_path}: got {len(encoded_list)}, expected {len(seen)}"
                )

            output_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_file.with_suffix(output_file.suffix + ".tmp")
            torch.save(encoded_list, tmp_path)
            os.replace(tmp_path, output_file)
            logger.info(f"Saved {len(encoded_list)} embeddings -> {output_file}")
            return True

        except Exception as exc:
            logger.error(f"Error processing {json_path}: {exc}", exc_info=True)
            return False


def process_t5_batch(args):
    processor, file_pairs = args

    if processor.device.startswith("cuda:"):
        device_num = processor.device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_num
        processor.device = "cuda:0"
    else:
        processor.device = "cpu"

    results = []
    for json_path, output_path in file_pairs:
        success = processor.process_seen_instructions(
            json_path=json_path,
            output_path=output_path,
        )
        results.append((json_path, success))
    return results


def main():
    args = parse_args()
    wan_model_dir = resolve_wan_model_dir(args.wan_path)
    devices = parse_device_list(args.devices)

    logger.info(f"Dataset roots: {args.dataset_roots}")
    logger.info(f"Resolved WAN model dir: {wan_model_dir}")
    logger.info(f"Devices: {devices}")
    logger.info(f"Output filename: {DEFAULT_OUTPUT_NAME}")

    pairs, skipped_existing, skipped_invalid = collect_episode_pairs(
        dataset_roots=args.dataset_roots,
    )

    logger.info(f"Episodes queued: {len(pairs)}")
    logger.info(f"Skipped existing: {skipped_existing}")
    logger.info(f"Skipped invalid: {skipped_invalid}")

    if not pairs:
        logger.info("Nothing to process.")
        return

    chunks = [pairs[i::len(devices)] for i in range(len(devices))]
    chunks = [chunk for chunk in chunks if chunk]

    processors_and_chunks = []
    for device, chunk in zip(devices, chunks):
        processor = T5EmbeddingProcessor(
            wan_model_dir=str(wan_model_dir),
            text_len=DEFAULT_TEXT_LEN,
            device=device,
        )
        processors_and_chunks.append((processor, chunk))

    if len(processors_and_chunks) == 1:
        all_results = process_t5_batch(processors_and_chunks[0])
    else:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(processors_and_chunks),
            mp_context=mp_context,
        ) as executor:
            futures = [executor.submit(process_t5_batch, item) for item in processors_and_chunks]
            all_results = []
            for future in tqdm(futures, desc="T5 embedding workers"):
                all_results.extend(future.result())

    successful = sum(1 for _, ok in all_results if ok)
    total = len(all_results)
    logger.info(f"Finished: {successful}/{total} episodes successful")
    if successful < total:
        failed = [path for path, ok in all_results if not ok]
        logger.warning(f"Failed episodes ({len(failed)}): {failed[:20]}")


if __name__ == "__main__":
    main()
