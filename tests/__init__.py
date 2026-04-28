# V3 tests
import sys as _sys
from pathlib import Path as _Path
_root = str(_Path(__file__).resolve().parents[1])
if _root not in _sys.path:
    _sys.path.insert(0, _root)
