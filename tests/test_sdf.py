from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.map_utils import Workspace
from src.sdf_utils import compute_sdf


def test_sdf_signs():
    ws = Workspace((0.0, 1.0), (0.0, 1.0), 50)
    X, Y = ws.meshgrid()
    mask = ((X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.12**2).astype(np.uint8)
    sdf = compute_sdf(mask, ws)
    assert sdf[25, 25] < 0.0
    assert sdf[5, 5] > 0.0
