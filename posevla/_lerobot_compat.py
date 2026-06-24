# Copyright 2026 PoseVLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Compatibility shim that centralizes lerobot version-dependent imports.

`lerobot` keeps re-shuffling its package layout across releases (0.1.0 used
`lerobot.common.*`, 0.4.2 moved everything to `lerobot.*` and renamed a few
sub-packages). Multiple files in this repository used to repeat the same
``try / if version == ...`` dance, which made things noisy and easy to break.

This module performs the version probe **once** and re-exports a stable set
of symbols that downstream code can simply do::

    from posevla._lerobot_compat import (
        LEROBOT_VERSION,
        PreTrainedPolicy,
        get_safe_dtype,
        ACTION, OBS_IMAGES, OBS_STATE,
        AdamWConfig, CosineDecayWithWarmupSchedulerConfig,
        LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset,
        ImageTransforms, ImageTransformsConfig,
        StreamingLeRobotDataset,  # may be None on old versions
        resolve_delta_timestamps,  # may be None on new versions
    )

Symbols that genuinely do not exist on a given lerobot version are exposed as
``None`` rather than raising at import time, so that callers can gracefully
opt out (e.g. streaming datasets only exist in >= 0.4.2).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import lerobot

# ---------------------------------------------------------------------------
# 1. Version probe (printed once)
# ---------------------------------------------------------------------------
try:
    LEROBOT_VERSION: str = lerobot.__version__
except AttributeError:
    LEROBOT_VERSION = "unknown"

print(f"[lerobot_compat] Detected lerobot version: {LEROBOT_VERSION}")

_IS_LEGACY = LEROBOT_VERSION == "0.1.0"   # `lerobot.common.*` layout
_IS_NEW = LEROBOT_VERSION == "0.4.2"      # flat `lerobot.*` layout


# ---------------------------------------------------------------------------
# 2. PreTrainedPolicy
#    On lerobot 0.4.2 `lerobot.policies/__init__.py` is a heavy module that
#    eagerly imports many extra dependencies we do not need. We bypass it by
#    injecting a stub package into sys.modules so that
#    `import lerobot.policies.pretrained` skips the real __init__.py.
# ---------------------------------------------------------------------------
if _IS_LEGACY:
    from lerobot.common.policies.pretrained import PreTrainedPolicy  # noqa: F401
elif _IS_NEW:
    _policy_path = Path(lerobot.__file__).parent / "policies"
    _stub = types.ModuleType("lerobot.policies")
    _stub.__path__ = [str(_policy_path)]
    sys.modules.setdefault("lerobot.policies", _stub)

    import lerobot.policies.pretrained as _pretrained  # noqa: E402

    PreTrainedPolicy = _pretrained.PreTrainedPolicy
    print("[lerobot_compat] Bypassed lerobot.policies.__init__")
else:
    # Best-effort fallback for unknown versions.
    try:
        from lerobot.policies.pretrained import PreTrainedPolicy  # type: ignore[no-redef]  # noqa: F401
    except ImportError:
        from lerobot.common.policies.pretrained import PreTrainedPolicy  # type: ignore[no-redef]  # noqa: F401


# ---------------------------------------------------------------------------
# 3. Constants & generic utils
# ---------------------------------------------------------------------------
if _IS_LEGACY:
    from lerobot.common.utils.utils import get_safe_dtype  # noqa: F401
    from lerobot.common.constants import ACTION, OBS_IMAGES  # noqa: F401
    from lerobot.common.constants import OBS_ROBOT as OBS_STATE  # noqa: F401
else:
    try:
        from lerobot.utils.utils import get_safe_dtype  # noqa: F401
        from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: F401
    except ImportError:  # pragma: no cover - last-resort fallback
        from lerobot.common.utils.utils import get_safe_dtype  # type: ignore[no-redef]  # noqa: F401
        from lerobot.common.constants import ACTION, OBS_IMAGES  # type: ignore[no-redef]  # noqa: F401
        from lerobot.common.constants import OBS_ROBOT as OBS_STATE  # type: ignore[no-redef]  # noqa: F401


# ---------------------------------------------------------------------------
# 4. Optimizer / scheduler configs
# ---------------------------------------------------------------------------
if _IS_LEGACY:
    from lerobot.common.optim.optimizers import AdamWConfig  # noqa: F401
    from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig  # noqa: F401
else:
    try:
        from lerobot.optim.optimizers import AdamWConfig  # noqa: F401
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig  # noqa: F401
    except ImportError:  # pragma: no cover
        from lerobot.common.optim.optimizers import AdamWConfig  # type: ignore[no-redef]  # noqa: F401
        from lerobot.common.optim.schedulers import (  # type: ignore[no-redef]  # noqa: F401
            CosineDecayWithWarmupSchedulerConfig,
        )


# ---------------------------------------------------------------------------
# 5. Datasets / transforms
#    `resolve_delta_timestamps` exists only in 0.1.0; `StreamingLeRobotDataset`
#    only in 0.4.2. We expose them as `None` on the version that lacks them so
#    callers can do `if StreamingLeRobotDataset is not None:` checks.
# ---------------------------------------------------------------------------
StreamingLeRobotDataset = None
resolve_delta_timestamps = None

if _IS_LEGACY:
    from lerobot.common.datasets.lerobot_dataset import (  # noqa: F401
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
    )
    from lerobot.common.datasets.transforms import ImageTransforms, ImageTransformsConfig  # noqa: F401
    from lerobot.common.datasets.factory import resolve_delta_timestamps  # noqa: F401
else:
    try:
        from lerobot.datasets.lerobot_dataset import (  # noqa: F401
            LeRobotDataset,
            LeRobotDatasetMetadata,
            MultiLeRobotDataset,
        )
        from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig  # noqa: F401
        try:
            from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: F401
        except ImportError:
            resolve_delta_timestamps = None
        try:
            from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset  # noqa: F401
        except ImportError:
            StreamingLeRobotDataset = None
    except ImportError:  # pragma: no cover
        from lerobot.common.datasets.lerobot_dataset import (  # type: ignore[no-redef]  # noqa: F401
            LeRobotDataset,
            LeRobotDatasetMetadata,
            MultiLeRobotDataset,
        )
        from lerobot.common.datasets.transforms import (  # type: ignore[no-redef]  # noqa: F401
            ImageTransforms,
            ImageTransformsConfig,
        )


__all__ = [
    "LEROBOT_VERSION",
    # policy
    "PreTrainedPolicy",
    # utils & constants
    "get_safe_dtype",
    "ACTION",
    "OBS_IMAGES",
    "OBS_STATE",
    # optim
    "AdamWConfig",
    "CosineDecayWithWarmupSchedulerConfig",
    # datasets
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
    "ImageTransforms",
    "ImageTransformsConfig",
    "StreamingLeRobotDataset",
    "resolve_delta_timestamps",
]