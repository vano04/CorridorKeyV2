from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import infer
from Infer.inference import Tile, _save_frame_outputs, _tile_weight


def test_v3_infer_defaults_to_model_fg_outputs():
    args = infer.parse_args(["--checkpoint", "dummy.pt"])
    assert args.fg_source == "model"


def test_global_context_letterbox_centers_wide_frames():
    frame = torch.ones((3, 8, 16), dtype=torch.float32)

    out = infer._letterbox_chw_to_square(frame, 16, fill=0.0)

    assert tuple(out.shape) == (3, 16, 16)
    assert torch.count_nonzero(out[:, :4]) == 0
    assert torch.count_nonzero(out[:, 12:]) == 0
    assert torch.allclose(out[:, 4:12], torch.ones((3, 8, 16)))


def test_global_fg_guidance_masked_input_uses_seed_alpha():
    video = torch.ones((2, 3, 4, 4), dtype=torch.float32)
    seed = torch.zeros((1, 4, 4), dtype=torch.float32)
    seed[:, 1:3, 2:] = 0.5

    out = infer._build_pseudo_global_fg_guidance(video, seed, "masked-input")

    assert tuple(out.shape) == (2, 3, 4, 4)
    assert torch.count_nonzero(out[:, :, :, :2]) == 0
    assert torch.allclose(out[:, :, 1:3, 2:], torch.full((2, 3, 2, 2), 0.5))


def test_tile_weight_uses_half_overlap_linear_ramp():
    tile = Tile(y0=64, y1=320, x0=64, x1=320)

    weight = _tile_weight(tile, full_h=384, full_w=384, overlap=64, device=torch.device("cpu"))

    assert tuple(weight.shape) == (1, 256, 256)
    assert 0.0 < float(weight[:, 0, 128]) < float(weight[:, 15, 128]) < float(weight[:, 30, 128]) < 1.0
    assert 0.0 < float(weight[:, 128, 0]) < float(weight[:, 128, 15]) < float(weight[:, 128, 30]) < 1.0
    assert torch.allclose(weight[:, 128, 128], torch.ones(()))


def test_tile_core_bounds_split_overlap_at_midpoint():
    tile = Tile(y0=768, y1=1792, x0=768, x1=1792)

    core = infer._tile_core_bounds(
        tile=tile,
        y_starts=[0, 768, 1536],
        x_starts=[0, 768, 1536],
        tile_size=1024,
        full_h=2560,
        full_w=2560,
    )

    assert core == Tile(y0=896, y1=1664, x0=896, x1=1664)


def test_premul_model_fg_preview_does_not_unpremultiply_tiny_alpha(tmp_path):
    alpha = torch.full((1, 2, 2), 1e-4, dtype=torch.float32)
    fg = torch.zeros((3, 2, 2), dtype=torch.float32)
    fg[1] = 1e-3
    fg[2] = 1e-3

    _save_frame_outputs(
        index=0,
        alpha=alpha,
        fg=fg,
        input_path=None,
        input_info=None,
        output_dir=tmp_path,
        fg_representation="premul",
        fg_source="model",
    )

    fg_png = np.asarray(Image.open(tmp_path / "FG" / "fg_00000.png"))
    comp_png = np.asarray(Image.open(tmp_path / "Comp" / "comp_00000.png"))

    assert fg_png[..., 1].max() < 16
    assert fg_png[..., 2].max() < 16
    assert comp_png[..., 1].max() < 160
    assert comp_png[..., 2].max() < 160
