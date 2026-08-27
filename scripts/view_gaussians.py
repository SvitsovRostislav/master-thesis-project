"""Entry point: launch the base 3D Gaussian Splatting viewer.

Usage:
    .venv\\Scripts\\python scripts\\view_gaussians.py
    .venv\\Scripts\\python scripts\\view_gaussians.py data\\raw\\test_models\\sample_gaussians.ply

With no argument, a file picker dialog opens in data/raw/test_models/.
Controls: left-drag = rotate, scroll = zoom, right-drag = pan, O = open
another file, Esc = quit.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # allow `from src...` when run as a plain script

from src.viewer.app import main  # noqa: E402

if __name__ == "__main__":
    main()
