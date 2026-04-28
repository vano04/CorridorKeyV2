"""CorridorKeyV2 training entrypoint.

Swaps the V3 model builder and loss computer into the root trainer via the
same monkey-patch pattern used by V2. The root ``train()`` function handles
DDP, checkpointing, scheduling, and data loading unchanged.
"""
from __future__ import annotations

try:
    import transformer_engine.pytorch as te  # noqa: F401  # Fix library loading order for cublasLt when available
except Exception:
    te = None  # type: ignore[assignment]
import sys
from pathlib import Path

# Make the project root importable regardless of the working directory, and
# also keep the mono-repo root available for shared helpers.
_project_root = str(Path(__file__).resolve().parent)
_repo_root = str(Path(__file__).resolve().parent.parent)
for _path in (_project_root, _repo_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import train as root_train  # noqa: E402

from models import build_v3_hybrid_video_matting_model  # noqa: E402
from losses import V3MattingLossComputer  # noqa: E402


# ---------------------------------------------------------------------------
# Monkey-patch the root trainer to use V3 model and loss.
#
# ``root_train.train()`` calls:
#   model = build_memory_guided_video_matting_model(model_cfg)
#   criterion = MattingLossComputer(weights, fg_representation)
#
# We replace both callables so the rest of the training runtime (DDP,
# checkpointing, scheduling, data loading, CUDA warmup, etc.) is reused
# without modification.
# ---------------------------------------------------------------------------

root_train.build_memory_guided_video_matting_model = build_v3_hybrid_video_matting_model  # type: ignore[attr-defined]
root_train.MattingLossComputer = V3MattingLossComputer  # type: ignore[attr-defined]


if __name__ == "__main__":
    root_train.train()
