from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_async_prefetcher_slow_consumer_regression_completes_in_subprocess() -> None:
    code = textwrap.dedent(
        """
        import time
        import torch

        from utils.prefetch import AsyncDevicePrefetcher

        prefetcher = AsyncDevicePrefetcher(
            dataloader=[{"x": torch.tensor([0])}, {"x": torch.tensor([1])}],
            device=torch.device("cpu"),
            queue_size=1,
        )
        iterator = iter(prefetcher)
        assert int(next(iterator)["x"].item()) == 0
        time.sleep(1.2)
        assert int(next(iterator)["x"].item()) == 1
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise AssertionError("prefetcher yielded after exhaustion")
        """
    )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=8.0,
    )
    assert result.returncode == 0, result.stderr
