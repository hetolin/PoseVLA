"""
Dataset / DataLoader factory for PoseVLA.

This module centralizes the construction of VLM and Action datasets so that
``train.py`` and ``pi0/modeling_pi0.py::test`` (and any future entry point)
share the same logic.

Public API
----------
- build_weighted_sampler(datasets, alpha=0.43)
    Concatenate multiple datasets and return ``(concat_dataset, sampler, weights)``
    using the ``n ** alpha`` rule.

- build_vlm_dataloaders(cfg, pi0_config, bin_tokenizer)
    Build train/val VLM dataloaders. Picks the 3D-detection branch
    (``cfg.data_3d == True``) or the agibot + interndata-a1 branch.

- build_action_dataloader(cfg, pi0_config)
    Build the action-training dataloader for the ``hdf5`` or ``lerobot`` format.
"""
from typing import List, Optional, Sequence, Tuple

import hydra
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def build_weighted_sampler(
    datasets: Sequence[Dataset],
    alpha: float = 0.43,
    verbose: bool = True,
) -> Tuple[ConcatDataset, WeightedRandomSampler, List[float]]:
    """Concatenate ``datasets`` and build a WeightedRandomSampler using the
    ``n ** alpha`` rule, where ``n`` is the length of each sub-dataset.

    Returns
    -------
    concat_dataset : ConcatDataset
    sampler        : WeightedRandomSampler
    weights        : list[float]  per-sample weights (mainly for debugging)
    """
    lengths = [len(ds) for ds in datasets]
    combo_weights = [n ** alpha for n in lengths]

    # Per-sample weight = (combo_weight / dataset_len) so that the total
    # weight of each dataset equals combo_weight (= n ** alpha).
    weights: List[float] = []
    for dataset_weight, dataset_len in zip(combo_weights, lengths):
        weights += [dataset_weight / dataset_len] * dataset_len

    if verbose:
        sum_total = sum(combo_weights)
        print("Sampling ratio per dataset:")
        for idx, (l, tw) in enumerate(zip(lengths, combo_weights)):
            print(f"  dataset {idx} (length={l}), sampling ratio={tw / sum_total:.4f}")

    concat_dataset = ConcatDataset(list(datasets))
    sampler = WeightedRandomSampler(
        weights, num_samples=sum(lengths), replacement=True
    )
    return concat_dataset, sampler, weights


# --------------------------------------------------------------------------- #
# VLM branch
# --------------------------------------------------------------------------- #
def _build_vlm_3d(cfg, bin_tokenizer):
    """Build VLM dataloaders for the 3D detection / pose branch."""
    # Local imports keep the module light when this branch is not used.
    from data.collators import CollatorForDetectionDataset
    from data.ds_train.dataset_bop import BopConsumerDataset
    from data.ds_train.dataset_clutter import BopClutterConsumerDataset
    from data.ds_train.dataset_omni3d import Omni3DConsumerDataset
    from data.ds_train.dataset_omni6d import Omni6dConsumerDataset

    omni6d_dataset = Omni6dConsumerDataset(config=cfg, tokenizer=bin_tokenizer)
    omni3d_dataset = Omni3DConsumerDataset(
        config=cfg, tokenizer=bin_tokenizer, yaml_name="omni3d_train"
    )
    bop_train_pbr_dataset = BopConsumerDataset(
        config=cfg, tokenizer=bin_tokenizer, yaml_name="bop_train_pbr"
    )
    bop_test_dataset = BopConsumerDataset(
        config=cfg, tokenizer=bin_tokenizer, yaml_name="bop_test"
    )
    clutter_train_dataset = BopClutterConsumerDataset(
        config=cfg, tokenizer=bin_tokenizer, yaml_name="clutter_train"
    )
    clutter_test_dataset = BopClutterConsumerDataset(
        config=cfg, tokenizer=bin_tokenizer, yaml_name="clutter_test"
    )

    sub_datasets = [
        omni6d_dataset,
        omni3d_dataset,
        bop_train_pbr_dataset,
        bop_test_dataset,
        clutter_train_dataset,
        clutter_test_dataset,
    ]

    sampler: Optional[WeightedRandomSampler]
    if cfg.training.weighted_sample:
        train_vlm_dataset, sampler, _ = build_weighted_sampler(sub_datasets)
        cfg.dataloader.shuffle = False
    else:
        train_vlm_dataset = ConcatDataset(sub_datasets)
        sampler = None

    data_vlm_collator = CollatorForDetectionDataset(
        config=cfg, sub_cfg_key="dataset_det"
    )
    train_vlm_dataloader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=train_vlm_dataset,
        collate_fn=data_vlm_collator,
        sampler=sampler,
    )

    # Validation only uses omni3d_val
    omni3d_val_dataset = Omni3DConsumerDataset(
        config=cfg, tokenizer=bin_tokenizer, yaml_name="omni3d_val"
    )
    val_vlm_dataset = ConcatDataset([omni3d_val_dataset])
    val_vlm_dataloader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=val_vlm_dataset,
        collate_fn=data_vlm_collator,
        sampler=None,
    )

    return train_vlm_dataloader, val_vlm_dataloader, train_vlm_dataset


def _build_vlm_robot(cfg, pi0_config, bin_tokenizer):
    """Build VLM dataloaders for the agibot + interndata-a1 branch."""
    from data.ds_raw.interndata_a1 import load_interndata_a1
    from data.ds_train.dataset_lerobot import make_dataset
    from data.ds_train.dataset_agibot import AgibotConsumerDataset, DataCollatorForAgibotConsumerDataset

    agibot_data = AgibotConsumerDataset(
        config=cfg,
        tokenizer=bin_tokenizer,
        action_type="ee",
        future_video=False,
        global_downsample_rate=3,
    )

    yaml_names = [
        "lerobot_group01",
        "lerobot_group02",
        "lerobot_group03",
        "lerobot_group04",
        "lerobot_group06",
    ]
    interdataa1_data_list = load_interndata_a1(
        pi0_config, cfg, bin_tokenizer, make_dataset, yaml_names
    )

    train_vlm_dataset = ConcatDataset([agibot_data] + interdataa1_data_list)
    data_vlm_collator = DataCollatorForAgibotConsumerDataset(config=cfg)

    train_vlm_dataloader = hydra.utils.instantiate(
        cfg.dataloader, dataset=train_vlm_dataset, collate_fn=data_vlm_collator
    )
    # NOTE: validation reuses the train dataset here (mirrors original train.py).
    val_vlm_dataloader = hydra.utils.instantiate(
        cfg.dataloader, dataset=train_vlm_dataset, collate_fn=data_vlm_collator
    )
    return train_vlm_dataloader, val_vlm_dataloader, train_vlm_dataset


def build_vlm_dataloaders(cfg, pi0_config, bin_tokenizer):
    """Public entry: dispatch to the 3D or agibot VLM dataloader builder.

    Returns
    -------
    train_vlm_dataloader, val_vlm_dataloader, train_vlm_dataset
    """
    if cfg.data_3d:
        return _build_vlm_3d(cfg, bin_tokenizer)
    return _build_vlm_robot(cfg, pi0_config, bin_tokenizer)


# --------------------------------------------------------------------------- #
# Action branch
# --------------------------------------------------------------------------- #
def build_action_dataloader(cfg, pi0_config):
    """Build the action-training dataloader.

    Returns
    -------
    train_dataloader, train_dataset
    """
    if cfg.dataset.type == "hdf5":
        from data.collators import CollatorForActionConsumerDataset
        from data.ds_train.dataset_hdf5 import VLAConsumerDataset

        train_dataset = VLAConsumerDataset(config=cfg)
        data_collator = CollatorForActionConsumerDataset(config=cfg)
        train_dataloader = hydra.utils.instantiate(
            cfg.dataloader, dataset=train_dataset, collate_fn=data_collator
        )
    elif cfg.dataset.type == "lerobot":
        from data.ds_train.dataset_hdf5 import make_dataset

        train_dataset = make_dataset(pi0_config, cfg)
        train_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset)
    else:
        raise NotImplementedError(
            "Only 'hdf5' and 'lerobot' dataset formats are supported."
        )

    return train_dataloader, train_dataset