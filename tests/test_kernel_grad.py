from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.kernels import grad_x_rbf_kernel, rbf_kernel


def test_kernel_gradient_matches_finite_difference():
    x = np.array([[0.2, 0.3]])
    y = np.array([[0.4, 0.6]])
    h = 0.1

    g = grad_x_rbf_kernel(x, y, h)[0, 0, :]

    eps = 1e-6
    num = np.zeros(2)
    for d in range(2):
        x1 = x.copy()
        x2 = x.copy()
        x1[0, d] += eps
        x2[0, d] -= eps
        f1 = rbf_kernel(x1, y, h)[0, 0]
        f2 = rbf_kernel(x2, y, h)[0, 0]
        num[d] = (f1 - f2) / (2 * eps)
    assert np.allclose(g, num, atol=1e-5)
