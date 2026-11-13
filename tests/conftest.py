"""Shared test setup.

The scripts in src/ import each other by module name (for example
`from scunet_model import SCUNet`), so src/ must be on the import path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
