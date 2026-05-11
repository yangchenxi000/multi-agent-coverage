import numpy as np

from src.trajectory_update import update_trajectory_points


def test_trajectory_update_keeps_start_and_end_when_fixed():
    traj = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
    direction = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)
    out = update_trajectory_points(traj, direction, step_size=0.1, smooth_lambda=0.0, fix_start=True, fix_end=True)
    assert np.allclose(out[0], traj[0])
    assert np.allclose(out[-1], traj[-1])
    assert out[1, 1] > traj[1, 1]
