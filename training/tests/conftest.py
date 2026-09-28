"""Make `import atlas_data` / `import atlas_train` work when pytest runs from the repo root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
