from __future__ import annotations

from torch import Tensor


FG_REPRESENTATIONS = frozenset({"premul", "straight"})


def validate_fg_representation(value: str) -> str:
    rep = str(value).strip().lower()
    if rep not in FG_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported fg_representation={value!r}. "
            f"Expected one of: {', '.join(sorted(FG_REPRESENTATIONS))}."
        )
    return rep


def fg_target_from_premul(fg_premul: Tensor, alpha: Tensor, representation: str, eps: float = 1e-3) -> Tensor:
    """Return the supervised foreground target for premul or straight training."""
    if representation == "premul":
        return fg_premul

    alpha_safe = alpha.clamp_min(eps)
    visible = (alpha > eps).to(fg_premul.dtype)
    return (fg_premul / alpha_safe) * visible


def composite_fg(fg: Tensor, bg: Tensor, alpha: Tensor, representation: str) -> Tensor:
    if representation == "straight":
        return fg * alpha + (1.0 - alpha) * bg
    return fg + (1.0 - alpha) * bg
