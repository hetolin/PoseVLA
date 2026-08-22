"""Launch the isolated PI0.5 baseline through PoseVLA's shared trainer."""

import os
import sys
from inspect import signature
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parent
POSEVLA_ROOT = BASELINE_ROOT.parent


def main() -> None:
    os.chdir(POSEVLA_ROOT)
    if str(POSEVLA_ROOT) not in sys.path:
        sys.path.insert(0, str(POSEVLA_ROOT))

    import train_posttrain
    from accelerate import Accelerator
    from pi05_baseline.robotwin_eval.model import PI05Policy

    if "dispatch_batches" not in signature(Accelerator).parameters:
        from accelerate import DataLoaderConfiguration

        def accelerator_compat(*args, dispatch_batches=None, **kwargs):
            kwargs["dataloader_config"] = DataLoaderConfiguration(
                dispatch_batches=dispatch_batches
            )
            return Accelerator(*args, **kwargs)

        train_posttrain.Accelerator = accelerator_compat

    # The current PoseVLA tokenizer has one extra token compared with the
    # released PI0.5 checkpoint. Reuse the eval policy's compatibility class
    # so the shared trainer preserves the checkpoint's embedding shape.
    train_posttrain.PoseVLAPolicy = PI05Policy
    sys.argv = [
        str(POSEVLA_ROOT / "train_posttrain.py"),
        f"--config-path={BASELINE_ROOT / 'config'}",
        "--config-name=train",
        *sys.argv[1:],
    ]
    train_posttrain.train()


if __name__ == "__main__":
    main()
