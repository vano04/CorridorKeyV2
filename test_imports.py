import sys
from pathlib import Path
_project_root = str(Path("/home/ivan/Corridor/CorridorKeyV2"))
sys.path.insert(0, _project_root)

import argparse
import importlib
# ... just import the main module
try:
    import training_web
except Exception as e:
    import traceback
    traceback.print_exc()
