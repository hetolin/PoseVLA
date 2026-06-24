from dataclasses import dataclass, field
from typing import ClassVar

from posevla._lerobot_compat import (
    AdamWConfig,
    CosineDecayWithWarmupSchedulerConfig,
    LEROBOT_VERSION as _LEROBOT_VERSION,  # re-exported in case downstream code reads it
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("pi0_ours")
@dataclass
class PI0Config(PreTrainedConfig):
    """PoseVLA / Pi0 policy configuration.

    Fields are grouped by purpose for readability:
        1. Temporal / sequence structure
        2. Normalization & feature definitions
        3. Dimension padding
        4. Image preprocessing
        5. Model architecture & tokenizer
        6. Fine-tuning strategy
        7. Optimizer & LR scheduler
        8. Advanced training / inference switches
    """

    # ------------------------------------------------------------------ #
    # 1. Temporal / sequence structure
    # ------------------------------------------------------------------ #
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # ------------------------------------------------------------------ #
    # 2. Normalization & feature definitions
    #    image_features / action_feature are class-level constants
    #    (not dataclass fields); ClassVar makes that explicit.
    # ------------------------------------------------------------------ #
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    image_features: ClassVar[dict[str, PolicyFeature]] = {
        "observation.images.top_head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.hand_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.hand_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }

    action_feature: ClassVar[PolicyFeature] = PolicyFeature(type=FeatureType.ACTION, shape=(14,))

    # ------------------------------------------------------------------ #
    # 3. Dimension padding
    #    Shorter state / action vectors will be zero-padded to these dims.
    # ------------------------------------------------------------------ #
    max_state_dim: int = 32
    max_action_dim: int = 32

    # ------------------------------------------------------------------ #
    # 4. Image preprocessing
    # ------------------------------------------------------------------ #
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Number of placeholder (empty) cameras to append. Used by pi0_aloha_sim
    # which adds empty left/right wrist cameras in addition to the top one.
    empty_cameras: int = 0

    # ------------------------------------------------------------------ #
    # 5. Model architecture & tokenizer
    # ------------------------------------------------------------------ #
    tokenizer_max_length: int = 512  # historical candidates: 256 / 96 / 48
    tokenizer_model_path: str = "google/paligemma-3b-pt-224"

    proj_width: int = 1024
    num_steps: int = 10  # flow-matching decoding steps

    # Attention backend: "eager", "fa2" or "flex".
    use_cache: bool = True
    attention_implementation: str = "eager"

    pi05: bool = True
    is_knowledge_insulation: bool = False

    add_extra_token: bool = False
    add_image_token: bool = False
    add_prior: bool = True

    # ------------------------------------------------------------------ #
    # 6. Fine-tuning strategy
    # ------------------------------------------------------------------ #
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True

    # ------------------------------------------------------------------ #
    # 7. Optimizer & LR scheduler
    # ------------------------------------------------------------------ #
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # ------------------------------------------------------------------ #
    # 8. Advanced training / inference switches
    # ------------------------------------------------------------------ #
    vis_attn: bool = False

    # Whether to skip random init of paligemma / gemma_expert weights.
    # Enable this only when the full weights will be immediately overwritten
    # by `from_pretrained`, which saves build time and peak memory.
    # Must remain False when training from scratch or only partially loading
    # pretrained weights.
    skip_init_weights: bool = False

    device: str = "cpu"
    # TODO: add EMA support.

    # ------------------------------------------------------------------ #
    # Validation & normalization
    # ------------------------------------------------------------------ #
    def __post_init__(self):
        super().__post_init__()

        # When deserialized from yaml/dict, tuple fields come back as lists.
        # Normalize them so downstream code can rely on tuple semantics.
        if isinstance(self.resize_imgs_with_padding, list):
            self.resize_imgs_with_padding = tuple(self.resize_imgs_with_padding)
        if isinstance(self.optimizer_betas, list):
            self.optimizer_betas = tuple(self.optimizer_betas)

        # ---- Basic field validation (non-exhaustive) ----
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The chunk size is the upper bound for the number of action steps per model invocation. "
                f"Got n_action_steps={self.n_action_steps} and chunk_size={self.chunk_size}."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got n_obs_steps={self.n_obs_steps}"
            )
        if self.attention_implementation not in {"eager", "fa2", "flex"}:
            raise ValueError(
                f"Invalid attention_implementation={self.attention_implementation!r}, "
                "expected one of {'eager', 'fa2', 'flex'}."
            )

    def validate_features(self) -> None:
        # Inject placeholder empty cameras into input_features so that the
        # data pipeline keeps a consistent number of visual streams.
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )

    # ------------------------------------------------------------------ #
    # Optimizer / scheduler presets
    # ------------------------------------------------------------------ #
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    # ------------------------------------------------------------------ #
    # Delta indices (required by the PreTrainedConfig interface)
    # ------------------------------------------------------------------ #
    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
