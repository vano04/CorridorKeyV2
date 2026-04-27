# V3 Hybrid Video Matting — Implementation Notes

## Overview

V3 is a three-branch hybrid video matting model built from scratch as a fresh
`nn.Module`. It processes 1024×1024 tiles at native resolution with global
context extracted from downscaled full frames.

## Architecture Summary

```
Input Frame (2048×2048)
    │
    ├─── Downsample ──► Global Context Branch (512×512)
    │                      │
    │                      ▼
    │                   GlobalContextBranch
    │                   4× Transformer blocks (spatial + temporal attention)
    │                   GlobalContextSummary (learnable query tokens)
    │                      │
    │                      ▼
    │                   global_tokens [B, M, C]
    │                      │
    ├─── Tile (1024²) ──►  │
    │                      ▼
    │                   Local Tile Encoder
    │                   Extended input: RGB(3) + hint(1) + flag(1) + green(2) + band(1) + coords(5)
    │                   4-stage transformer with:
    │                     • Window attention (per-frame)
    │                     • Temporal attention (cross-frame within memory window)
    │                     • Subject memory bank (sliding window)
    │                     • Global fusion cross-attention (from Branch 1)
    │                   FPN decoder → prediction heads
    │                      │
    │                      ▼
    │                   Prediction Heads
    │                   α, FG, uncertainty, spill, quality_eval
    │                      │
    │                      ▼
    │                   Native Detail Refiner (Branch 3)
    │                   Always-active soft-gated residuals
    │                   Input: RGB + coarse_α + coarse_FG + uncertainty + spill
    │                   Output: bounded deltas (tanh × max_delta × gate)
    │                      │
    │                      ▼
    └──────────────────► Final: α_refined, FG_refined, spill_refined
```

## Key Design Decisions

### 1. Fresh Module Architecture (no V1/V2 inheritance)
All modules are written from scratch. V2's conditional branching patterns
(which would break DDP `static_graph`) are not inherited.

### 2. DDP Static Graph Safety
Every parameter receives a gradient on every forward pass:
- **Native refiner**: always runs; uses soft gate (`refine_mask * learned_gate`) instead of conditional skip
- **Reference memory null_token**: added as a zero-weighted differentiable bias (`+ null_token.mean() * 0.0`) so the parameter is always in the graph regardless of dropout probability
- **Spill delta head**: always executes when `predict_spill=True`, using zeros as fallback input

### 3. Green/Chroma Priors
Two channels concatenated at patch-embedding level:
- `green_excess`: G - max(R, B), clamped [0, 1]
- `chroma_distance`: Euclidean distance in chroma space to reference green key

### 4. Hint Degradation
Applied inside `DeviceMattingTransform` after `coarse_alpha_init` generation:
- **clean** (25%): no degradation
- **mild** (50%): small morphological ops + blur + noise + shift
- **severe** (25%): large morphology + holes + threshold wobble + downsample-upsample + quadrant masking
Controlled by `hint_degrade_enabled` in config (default False for V1/V2 backward compat).

### 5. Coordinate Channels (5 channels)
Normalised tile position within the full frame:
- x0/W, y0/H, x1/W, y1/H (bounds)
- dist_to_edge (minimum normalised distance to frame boundary)

### 6. FP8 / Transformer Engine Strategy
- `MaybeTELinear` and `MaybeTELayerNorm` wrappers in `transformer_engine_utils.py`
- Fall back to standard PyTorch `nn.Linear`/`nn.LayerNorm` if TE is not available
- FP8 is for single-card RTX 5090 path only; DDP training is bf16-only
- Heads and native refiner are excluded from FP8

### 7. Loss Architecture (15 terms)

| Loss | Weight | Notes |
|------|--------|-------|
| alpha_l1 | 1.0 | Full-frame L1 |
| alpha_laplacian | 1.0 | Multi-scale Laplacian pyramid |
| fg_l1 | 1.0 | Foreground only where α > 0.01 |
| comp_l1 | 1.0 | Composite vs input |
| alpha_band_l1 | 0.5 | Transition region (0.02 < α < 0.98) |
| alpha_band_laplacian | 0.5 | Transition region pyramid |
| temporal_alpha_gradient | 0.3 | d/dt(pred) vs d/dt(target) |
| temporal_fg_gradient | 0.15 | FG temporal coherence |
| comp_random_bg | 0.3 | Composite over random BG |
| spill_l1 | 0.5 | Green spill mask BCE |
| coarse_alpha_l1 | 0.5 | Pre-refiner alpha supervision |
| coarse_fg_l1 | 0.3 | Pre-refiner FG supervision |
| native_alpha_delta_reg | 0.05 | Penalise large alpha deltas |
| native_fg_delta_reg | 0.05 | Penalise large FG deltas |
| quality_eval | 0.1 | Self-calibration (train only) |

## File Map

```
V3/
├── __init__.py
├── training.py                    # Training entrypoint (monkey-patch into root trainer)
├── infer.py                       # Inference CLI
├── models/
│   ├── __init__.py
│   ├── v3_hybrid_matting.py       # Main V3 model + builder
│   ├── global_context.py          # Branch 1: global context
│   ├── local_tile_encoder.py      # Branch 2: local tile encoder
│   ├── native_detail_refiner.py   # Branch 3: native detail refiner
│   ├── reference_memory.py        # Reference memory bank
│   ├── heads.py                   # Prediction heads (α, FG, uncertainty, spill, quality)
│   ├── positional.py              # Tile coordinate utilities
│   └── transformer_engine_utils.py # TE FP8 wrappers
├── data/
│   ├── __init__.py
│   ├── green_priors.py            # Green/chroma prior maps
│   └── hint_degradation.py        # Hint degradation transform
├── losses/
│   ├── __init__.py
│   └── v3_matting_losses.py       # V3 loss computer (15 terms)
├── inference/
│   ├── __init__.py
│   ├── v3_tiled_runner.py         # Tile position + overlap merge
│   └── context_cache.py           # Cross-window token caching
├── utils/
│   ├── __init__.py
│   └── timing.py                  # Performance timing hooks
├── configs/
│   ├── v3_debug_smoke.yaml        # Debug: tiny dims, 5 steps
│   ├── v3_ddp_1024_hybrid.yaml    # Production: 4x 3090 DDP bf16
│   └── v3_single_5090_fp8.yaml    # Production: single RTX 5090 FP8
└── tests/
    ├── __init__.py
    ├── test_v3_shapes.py           # Output shape verification
    ├── test_v3_static_graph.py     # DDP gradient flow verification
    ├── test_v3_loss_smoke.py       # Loss forward+backward
    └── test_v3_inference_tile_merge.py  # Tile merge correctness

Modified files:
└── utils/device_transform.py      # Added hint degradation hook
```

## Integration with Root Trainer

V3 uses the same monkey-patch pattern as V2:

```python
# CorridorKeyV2/training.py
import train as root_train
root_train.build_memory_guided_video_matting_model = build_v3_hybrid_video_matting_model
root_train.MattingLossComputer = V3MattingLossComputer
root_train.train()
```

The root trainer handles DDP, checkpointing, scheduling, and data loading.
V3 only replaces the model builder and loss computer.

## Running

### Smoke Test (synthetic data)
```bash
python -c "
import torch
from models import build_v3_hybrid_video_matting_model
model = build_v3_hybrid_video_matting_model({...})
..."
```

### Unit Tests
```bash
python -m pytest V3/tests/ -v
```

### Training (requires CorridorKey dataset)
```bash
# Debug
python training.py --config configs/v3_debug_smoke.yaml --output-dir /tmp/v3_smoke

# DDP (4 GPUs)
torchrun --nproc_per_node=4 training.py --config configs/v3_ddp_1024_hybrid.yaml

# Single 5090
python training.py --config configs/v3_single_5090_fp8.yaml
```

### Inference
```bash
python infer.py \
    --checkpoint runs/v3_hybrid/checkpoint_best.pt \
    --input-dir Infer/demo_dwab1024/Input \
    --alpha-dir Infer/demo_dwab1024/Alpha \
    --output-dir V3_output \
    --tile-size 1024 --temporal-frames 4 --make-video
```
