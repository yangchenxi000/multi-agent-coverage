from src.density_utils import normalize_density
from src.map_utils import Workspace
import numpy as np


def test_density_norm():
    w = Workspace((0.0, 1.0), (0.0, 1.0), 50)
    q = np.ones((50, 50), dtype=float)
    qn = normalize_density(q, w)
    assert abs(float(np.sum(qn) * w.dx * w.dy) - 1.0) < 1e-6
