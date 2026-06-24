# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""Patch embedding modules used by the vision backbone.

The classes below are gathered from several upstream projects (croco, dust3r,
pow3r). Provenance markers such as ``# from <module> import <name>`` are kept
to make it easy to cross-reference the original implementations.
"""

import collections.abc
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


###########################################################
# from croco.models.blocks import Mlp
def _ntuple(n):
    """Return a parser that broadcasts a scalar into an ``n``-tuple."""

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP block as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


###########################################################
# from croco.models.blocks import PatchEmbed
class PositionGetter(object):
    """Return (and cache) the (y, x) positions of every patch in a feature map."""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if (h, w) not in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            # Cartesian product yields a (h * w, 2) tensor of (y, x) coordinates.
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos


class PatchEmbed(nn.Module):
    """Conv-based patch embedding.

    Compared with ``timm.models.layers.patch_embed.PatchEmbed`` this version
    additionally exposes ``_init_weights`` and a ``PositionGetter``.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.position_getter = PositionGetter()

    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


###########################################################
# from dust3r.patch_embed import PatchEmbedDust3R, ManyAR_PatchEmbed  # noqa
class PatchEmbedDust3R(PatchEmbed):
    """PatchEmbed variant that allows arbitrary ``H`` / ``W`` as long as both
    are multiples of the patch size (no fixed ``img_size`` check)."""

    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, (
            f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        )
        assert W % self.patch_size[1] == 0, (
            f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        )
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class ManyAR_PatchEmbed(PatchEmbed):
    """Patch embedding that handles non-square aspect ratios.

    All images in the same batch must share the same aspect ratio.
    ``true_shape = [(height, width), ...]`` carries each image's actual shape
    so that landscape and portrait inputs can be projected consistently.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f"img should be in landscape mode, but got {W=} {H=}"
        assert H % self.patch_size[0] == 0, (
            f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        )
        assert W % self.patch_size[1] == 0, (
            f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        )
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"

        # Sizes expressed in number of tokens.
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = width >= height
        is_portrait = ~is_landscape

        # Allocate output tensors.
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        # Linear projection, transposed for portrait images so that the
        # spatial axes are aligned with the landscape ones.
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()

        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        x = self.norm(x)
        return x, pos


###########################################################
# from pow3r.model.patch_embed import PatchEmbed_Mlp
class Permute(torch.nn.Module):
    """Module wrapper around ``torch.Tensor.permute`` so it can sit inside
    an ``nn.Sequential``."""

    dims: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = tuple(dims)

    def __repr__(self):
        return f"Permute{self.dims}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(*self.dims)


class PixelUnshuffle(nn.Module):
    """``nn.PixelUnshuffle`` wrapper that also supports empty tensors,
    which is useful when the batch contains zero samples on a given device."""

    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        if input.numel() == 0:
            # Branch not present in the original torch implementation.
            C, H, W = input.shape[-3:]
            assert H and W and H % self.downscale_factor == W % self.downscale_factor == 0
            return input.view(
                *input.shape[:-3],
                C * self.downscale_factor ** 2,
                H // self.downscale_factor,
                W // self.downscale_factor,
            )
        return F.pixel_unshuffle(input, self.downscale_factor)


class PatchEmbed_Mlp(PatchEmbedDust3R):
    """``PatchEmbedDust3R`` whose ``proj`` is an MLP applied on
    pixel-unshuffled patches (instead of a single Conv2d)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

        self.proj = nn.Sequential(
            PixelUnshuffle(patch_size),
            Permute((0, 2, 3, 1)),
            Mlp(in_chans * patch_size ** 2, 4 * embed_dim, embed_dim),
            Permute((0, 3, 1, 2)),
        )


class ManyAR_PatchEmbed_Mlp(ManyAR_PatchEmbed):
    """MLP-based variant of :class:`ManyAR_PatchEmbed`."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

        self.proj = nn.Sequential(
            PixelUnshuffle(patch_size),
            Permute((0, 2, 3, 1)),
            Mlp(in_chans * patch_size ** 2, 4 * embed_dim, embed_dim),
            Permute((0, 3, 1, 2)),
        )
