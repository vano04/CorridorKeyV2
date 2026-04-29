# CorridorKey EXR Data Analysis

Date: 2026-04-29

This note documents the local `CorridorKeyDataset` EXR shards as observed from
the current repository. It is meant as a data contract for future model and loss
work, especially around alpha scaling, HDR values, and premultiplied versus
straight foreground representations.

## Dataset Layout

The local dataset is stored as WebDataset-style tar shards:

- `CorridorKeyDataset/corridorkey-000000.tar` through `corridorkey-000033.tar`
- Each shard contains one JSON metadata file per clip plus per-frame EXRs.
- Example members from `corridorkey-000033.tar`:
  - `clip001056_shane_93.json`
  - `clip001056_shane_93.alpha.01.exr`
  - `clip001056_shane_93.fg.01.exr`
  - `clip001056_shane_93.bg.01.exr`
  - `clip001056_shane_93.input.01.exr`

The metadata records logical source paths such as `Alpha/01.exr`, `FG/01.exr`,
`BG/01.exr`, and `Input/01.exr`, with `modalities: ["alpha", "fg", "bg",
"input"]`.

## EXR Channel Format

Representative decoded EXRs were tiled OpenEXR files using DWAB compression.
The OpenEXR headers expose channels named `A`, `B`, `G`, and `R`.

The dataset loader composes these into CHW tensors in RGB(A) order:

- color EXRs become `[R, G, B, A]` if all four channels are present.
- training color tensors are reduced to RGB by `_ensure_color_three_channels`,
  which keeps only the first three channels.
- alpha EXRs can also arrive as 3 or 4 channels. Training alpha is reduced to a
  single channel by `_ensure_alpha_single_channel`, which keeps channel 0.

For the sampled `clip001056_shane_93.alpha.01.exr`, the alpha modality decoded
as `(4, 2048, 2048)`. Channels 0-2 were near-identical matte channels, while
channel 3 was constant 1.0. So the training choice of channel 0 is sensible for
these files; do not use the EXR `A` channel from the alpha modality as the matte
without checking the file convention.

## Raw Value Ranges

The raw EXRs are not all normalized LDR images.

Sampling 10 frames across 5 shards found:

- `alpha` is finite, but has small compression/ringing excursions:
  - minimum around `-9.3e-05`
  - maximum up to `1.078125`
  - some samples had up to about `8.7%` of alpha pixels above `1.0`
- `fg` is HDR in some clips:
  - maximum up to `29.046875`
  - nonzero fraction above `1.0` in HDR clips
- `bg` can be HDR:
  - maximum up to `3.837890625` in the sample
- `input` can be HDR:
  - maximum up to `27.0` in the sample

Small negative values occur in all modalities, consistent with compression or
filtering artifacts around zero. Models and losses should not assume raw disk
EXRs are already clamped to `[0, 1]`.

## Premultiplied Versus Straight

The current repository treats the source FG on disk as premultiplied. This is
also stated in `utils/data.py` near the representation conversion:

```python
# NOTE: ``fg_gt`` is the *premultiplied* foreground (FG = alpha * color)
# as it ships in the CorridorKey EXR dataset.
```

I compared the provided input plate against both matting equations:

- straight FG equation: `input ~= alpha * fg + (1 - alpha) * bg`
- premultiplied FG equation: `input ~= fg + (1 - alpha) * bg`

Across the sample, the premultiplied equation had lower median MAE:

- median straight MAE: `0.001411`
- median premultiplied MAE: `0.001007`

The difference is not enormous for all clips because many pixels are fully
transparent/opaque and some values are compressed or clipped, but the evidence
and code comments both point to this convention:

- **on disk `FG` is premultiplied HDR foreground**
- **on disk `Input` is the composited plate**
- **training may convert the supervised FG target to straight or premul,
  depending on `fg_representation`**

In the active configs inspected here, `fg_representation: premul`, so the model
is trained to predict premultiplied foreground after downstream clamping.

## Training Transform Contract

Raw disk format and training tensor format are different.

In `utils/data.py`, the transform first normalizes channels:

- color: keep first three channels
- alpha: keep first channel

Then it clamps alpha:

```python
alpha_gt = _ensure_alpha_single_channel(sample["Alpha"].to(torch.float32)).clamp(0.0, 1.0)
```

For device-offload training, this clamped alpha is returned to the training loop
and later consumed by `DeviceMattingTransform`. The device transform clamps
`input_gt`, `fg_gt`, and `bg_gt` after augmentation, while alpha has already
been made single-channel and clamped on the CPU side.

The training path also clamps the final supervised color targets and input into
`[0, 1]`:

- `input_gt.clamp_(0.0, 1.0)`
- `fg.clamp_(0.0, 1.0)`
- `bg.clamp_(0.0, 1.0)`

So current training is not directly optimizing against raw HDR color values. It
uses HDR raw EXRs as source material, applies representation conversion and
augmentation, then clamps the actual loss targets to the model's sigmoid output
range.

## Implications For Losses And Models

1. Loss code should assume `alpha_gt` in training batches is single-channel and
   clamped to `[0, 1]`, but raw EXR analysis tools must clamp/sanitize alpha
   themselves.
2. Loss code should not assume raw `FG`, `BG`, or `Input` on disk are LDR. They
   can be HDR and slightly negative.
3. If a future model predicts HDR foreground, the final sigmoid heads and target
   clamping need to be revisited. The current model/loss design is an LDR
   supervised target design, even though the source data is HDR.
4. If using straight foreground supervision, convert from premultiplied source
   carefully with alpha-safe division. Do not interpret disk `FG` as straight
   HDR color by default.
5. For debugging loss spikes, log both raw source stats and post-transform batch
   stats. A raw alpha max above 1 is expected in some files, but a post-transform
   `alpha_gt.max() > 1` would indicate a transform bug.

## Suggested Runtime Data Checks

For spike debugging, log this when `alpha_band_lap` or `total` exceeds a
threshold:

- `clip_id`
- `frame_indices`
- `alpha_gt.min/max/isfinite` after transform
- `alpha_pred.min/max/isfinite`
- `fg_gt.min/max`, `bg_gt.min/max`, `input_gt.min/max` after transform
- alpha band pixel count: `((alpha_gt > 0.02) & (alpha_gt < 0.98)).sum()`
- raw shard/member names if available

This separates true raw-data HDR behavior from invalid training tensors.
