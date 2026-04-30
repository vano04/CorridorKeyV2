"""Optional Weights & Biases logging helpers for training."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


class WandbLogger:
    """Thin wrapper that keeps wandb optional and rank-local."""

    def __init__(self, run: Any = None, *, log_checkpoints: bool = False) -> None:
        self.run = run
        self.log_checkpoints = log_checkpoints

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: Dict[str, Any], *, step: Optional[int] = None) -> None:
        if self.run is None:
            return
        self.run.log(metrics, step=step)

    def log_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        global_step: int,
        aliases: Optional[list[str]] = None,
    ) -> None:
        if self.run is None or not self.log_checkpoints:
            return
        import wandb

        artifact = wandb.Artifact(
            name=f"{self.run.id}-checkpoint",
            type="model",
            metadata={"epoch": int(epoch), "global_step": int(global_step)},
        )
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases or [f"step-{int(global_step)}"])

    def finish(self) -> None:
        if self.run is None:
            return
        self.run.finish()
        self.run = None


def init_wandb_logger(
    cfg: Dict[str, Any],
    *,
    output_dir: Path,
    rank0: bool,
    model: Any = None,
    debug_console: bool = False,
) -> WandbLogger:
    """Create a rank-0 wandb run from ``train.wandb`` config when enabled."""
    train_cfg = cfg.get("train", {})
    wandb_cfg = dict(train_cfg.get("wandb", {}) or {})
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled or not rank0:
        return WandbLogger()

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "train.wandb.enabled=true requires the 'wandb' package. "
            "Install it or set train.wandb.enabled=false."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = Path(wandb_cfg.get("dir", output_dir / "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs: Dict[str, Any] = {
        "project": wandb_cfg.get("project", "CorridorKeyV2"),
        "entity": wandb_cfg.get("entity", "vanoinc"),
        "config": cfg,
        "dir": str(wandb_dir),
        "resume": wandb_cfg.get("resume", "allow"),
    }
    for key in ("name", "group", "job_type", "notes", "tags", "mode", "id"):
        if key in wandb_cfg and wandb_cfg[key] is not None:
            init_kwargs[key] = wandb_cfg[key]

    if "mode" not in init_kwargs and os.environ.get("WANDB_MODE"):
        init_kwargs["mode"] = os.environ["WANDB_MODE"]

    run = wandb.init(**init_kwargs)
    if bool(wandb_cfg.get("watch_model", False)) and model is not None:
        watch_log = str(wandb_cfg.get("watch_log", "gradients"))
        watch_freq = int(wandb_cfg.get("watch_freq", 1000))
        wandb.watch(model, log=watch_log, log_freq=watch_freq)

    if debug_console:
        print(f"W&B logging enabled: project={init_kwargs['project']} run={run.name or run.id}", flush=True)

    return WandbLogger(
        run=run,
        log_checkpoints=bool(wandb_cfg.get("log_checkpoints", False)),
    )
