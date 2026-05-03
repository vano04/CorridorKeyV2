from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils.data import CorridorMattingTransform
from utils.device_transform import (
    DeviceMattingTransform,
    DeviceMattingTransformConfig,
    apply_green_foreground_augmentation,
)


def test_transform_sanitizes_hdr_and_invalid_source_values():
    transform = CorridorMattingTransform(
        clip_len_min=1,
        clip_len_max=1,
        resolution_buckets=(4,),
        horizontal_flip_p=0.0,
        background_replace_p=0.0,
        spill_augment_p=0.0,
        green_foreground_augment_p=0.0,
        temporal_jitter_p=0.0,
        subject_gain_min=1.0,
        subject_gain_max=1.0,
        bg_gain_min=1.0,
        bg_gain_max=1.0,
        wb_jitter_p=0.0,
        color_jitter_p=0.0,
        noise_p=0.0,
        shot_noise_p=0.0,
        blur_p=0.0,
        motion_blur_p=0.0,
        compression_p=0.0,
        fg_representation="premul",
    )

    fg = torch.zeros(1, 4, 4, 4)
    bg = torch.zeros_like(fg)
    alpha = torch.zeros(1, 4, 4, 4)

    fg[:, 0] = 159.0
    fg[:, 1, 0, 0] = float("inf")
    fg[:, 2, 0, 1] = float("nan")
    bg[:, 0, 0, 2] = -0.01
    alpha[:, 0] = 0.5
    alpha[:, 0, 0, 0] = float("nan")
    alpha[:, 0, 0, 1] = float("inf")
    alpha[:, 0, 0, 2] = -0.25
    alpha[:, 0, 0, 3] = 1.078125

    out = transform(
        {
            "FG": fg,
            "BG": bg,
            "Alpha": alpha,
            "clip_name": "synthetic_hdr_invalid",
            "frame_numbers": torch.tensor([1]),
        }
    )

    for key in ("alpha_gt", "fg_gt", "bg_gt", "video_rgb", "input_gt", "coarse_alpha_init"):
        assert torch.isfinite(out[key]).all(), key

    assert out["alpha_gt"].min() >= 0
    assert out["alpha_gt"].max() <= 1
    assert out["fg_gt"].min() >= 0
    assert out["fg_gt"].max() <= 1
    assert out["video_rgb"].min() >= 0
    assert out["video_rgb"].max() <= 1


def test_device_transform_sanitizes_offloaded_source_values():
    transform = DeviceMattingTransform(
        DeviceMattingTransformConfig(
            fixed_crop_size=0,
            horizontal_flip_p=0.0,
            background_replace_p=0.0,
            spill_augment_p=0.0,
            green_foreground_prob=0.0,
            subject_gain_min=1.0,
            subject_gain_max=1.0,
            bg_gain_min=1.0,
            bg_gain_max=1.0,
            wb_jitter_p=0.0,
            color_jitter_p=0.0,
            noise_p=0.0,
            shot_noise_p=0.0,
            blur_p=0.0,
            motion_blur_p=0.0,
            compression_p=0.0,
            grayscale_augment_p=0.0,
            fg_representation="premul",
        )
    )

    fg = torch.zeros(1, 1, 3, 4, 4)
    bg = torch.zeros_like(fg)
    alpha = torch.full((1, 1, 1, 4, 4), 0.5)
    fg[:, :, 0] = 159.0
    fg[:, :, 1, 0, 0] = float("inf")
    bg[:, :, 2, 0, 1] = float("nan")
    alpha[:, :, :, 0, 0] = float("nan")
    alpha[:, :, :, 0, 1] = float("inf")
    alpha[:, :, :, 0, 2] = -0.25
    alpha[:, :, :, 0, 3] = 1.078125

    out = transform({"fg_gt": fg, "bg_gt": bg, "alpha_gt": alpha})

    for key in ("alpha_gt", "fg_gt", "bg_gt", "video_rgb", "input_gt", "coarse_alpha_init", "alpha_boundary_gt"):
        assert torch.isfinite(out[key]).all(), key

    assert out["alpha_gt"].min() >= 0
    assert out["alpha_gt"].max() <= 1
    assert out["fg_gt"].min() >= 0
    assert out["fg_gt"].max() <= 1
    assert out["video_rgb"].min() >= 0
    assert out["video_rgb"].max() <= 1


def test_device_transform_sanitizes_precomputed_global_fg_guidance():
    transform = DeviceMattingTransform(
        DeviceMattingTransformConfig(
            fixed_crop_size=0,
            horizontal_flip_p=0.0,
            background_replace_p=0.0,
            spill_augment_p=0.0,
            green_foreground_prob=0.0,
            subject_gain_min=1.0,
            subject_gain_max=1.0,
            bg_gain_min=1.0,
            bg_gain_max=1.0,
            wb_jitter_p=0.0,
            color_jitter_p=0.0,
            noise_p=0.0,
            shot_noise_p=0.0,
            blur_p=0.0,
            motion_blur_p=0.0,
            compression_p=0.0,
            grayscale_augment_p=0.0,
            fg_representation="premul",
            global_context_long_side=4,
        )
    )

    fg = torch.zeros(1, 1, 3, 4, 4)
    bg = torch.zeros_like(fg)
    alpha = torch.full((1, 1, 1, 4, 4), 0.5)
    global_input = torch.full_like(fg, 0.25)
    global_alpha = torch.full_like(alpha, 0.5)
    global_fg = torch.zeros_like(fg)
    global_fg[:, :, 0] = 159.0
    global_fg[:, :, 1, 0, 0] = float("inf")
    global_fg[:, :, 2, 0, 1] = float("nan")

    out = transform(
        {
            "fg_gt": fg,
            "bg_gt": bg,
            "alpha_gt": alpha,
            "global_input_gt": global_input,
            "global_alpha_gt": global_alpha,
            "global_fg_gt": global_fg,
        }
    )

    assert torch.isfinite(out["global_video_rgb"]).all()
    assert torch.isfinite(out["global_fg_gt"]).all()
    assert torch.equal(out["global_video_rgb"], global_input)
    assert out["global_fg_gt"].min() >= 0
    assert out["global_fg_gt"].max() <= 1


def test_green_foreground_augmentation_preserves_alpha_and_shading():
    torch.manual_seed(0)
    alpha = torch.ones(1, 1, 1, 2, 2)
    straight = torch.tensor(
        [[[[[0.10, 0.20], [0.40, 0.80]], [[0.10, 0.20], [0.40, 0.80]], [[0.10, 0.20], [0.40, 0.80]]]]]
    )
    fg = straight * alpha

    out = apply_green_foreground_augmentation(
        fg,
        alpha,
        prob=1.0,
        strength_min=1.0,
        strength_max=1.0,
        alpha_thresh=0.75,
    )

    assert out.shape == fg.shape
    assert out[:, :, 1].mean() > out[:, :, 0].mean()
    assert out[:, :, 1].mean() > out[:, :, 2].mean()
    assert out[0, 0, 1, 0, 0] < out[0, 0, 1, 1, 1]

    low_alpha = alpha * 0.25
    unchanged = apply_green_foreground_augmentation(
        fg * low_alpha,
        low_alpha,
        prob=1.0,
        strength_min=1.0,
        strength_max=1.0,
        alpha_thresh=0.75,
    )
    assert torch.equal(unchanged, fg * low_alpha)
