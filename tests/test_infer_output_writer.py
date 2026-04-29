from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import infer
from Infer.inference import _save_frame_outputs


def test_v3_infer_defaults_to_model_fg_outputs():
    args = infer.parse_args(["--checkpoint", "dummy.pt"])
    assert args.fg_source == "model"


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
